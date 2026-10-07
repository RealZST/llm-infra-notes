#!/usr/bin/env python3
"""Real pretrained Qwen inference: unprofiled timings, separate trace/counter passes."""
# The harness that wrote data/raw/decode/ (mode performance) and the Python side of
# data/raw/counters/ (mode counters, run under Nsight Compute by ncu_counters.sh).
# Normally started by run_decode.sh, which prepares RUN with application.json,
# application_model.json (from prepare_model.py) and gpu_identity.json.
# Changes from the version used for the published runs, none of which touch the
# measurement: the model snapshot path comes from application_model.json instead of
# a fixed cluster path, and a GPU outside the three measured models needs > 15 GiB.
# The H100 and H200 runs used earlier revisions that differ only in out-of-memory
# handling (no case ran out of memory on those GPUs), the pilot-mode FP32 cache
# check, the trace-only `<phase>_model` profiler label (absent in the H200 decode and
# counter runs) and the recorded timing_protocol text. The timed loop and the
# counter capture are the same.
import argparse,contextlib,csv,functools,gc,hashlib,json,math,os,pathlib,sys,time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=pathlib.Path,required=True)
p.add_argument('--mode',choices=['pilot','performance','trace','counters'],required=True)
p.add_argument('--batch',type=int,default=1);p.add_argument('--length',type=int,default=1024)
p.add_argument('--phase',choices=['prefill','decode'],default='prefill')
p.add_argument('--layer',type=int,default=14)
a=p.parse_args();run=a.run;run.mkdir(parents=True,exist_ok=True)
config=json.loads((run/'application.json').read_text())
manifest=json.loads((run/'application_model.json').read_text())
snapshot=pathlib.Path(os.environ.get('APP_MODEL_SNAPSHOT',manifest['snapshot']))
for record in manifest['files']:
    f=snapshot/record['name'];st=f.stat()
    assert st.st_size==record['bytes'] and st.st_mtime_ns==record['mtime_ns'],'Cached model changed after hash verification'
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
assert torch.cuda.is_available() and torch.cuda.device_count()==1
prop=torch.cuda.get_device_properties(0)
identity=json.loads((run/'gpu_identity.json').read_text())
assert prop.name==identity['name']
# Minimum full-device memory per accepted GPU model (GiB); MIG slices excluded.
minimum=next((v for k,v in {'H200':130,'H100':70,'RTX 4080':15}.items() if k in prop.name),15)
assert 'MIG' not in prop.name and prop.total_memory>minimum*2**30,'Full GPU required, not MIG'
torch.manual_seed(config['seed']);torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
started=time.perf_counter()
print('Import complete; loading actual pretrained weights.',flush=True)
tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.bfloat16,
    attn_implementation=config['attention_implementation'],device_map={'':'cuda:0'}).eval()
assert model.config._attn_implementation=='sdpa'
assert len(model.model.layers)==28 and next(model.parameters()).dtype==torch.bfloat16
metadata={'mode':a.mode,'model_id':config['model_id'],'revision':config['revision'],'dtype':'bfloat16',
    'device':prop.name,'device_uuid':identity['uuid'],'memory_bytes':prop.total_memory,
    'torch':torch.__version__,'torch_cuda':torch.version.cuda,'transformers':transformers.__version__,
    'attention_implementation':model.config._attn_implementation,'cache':'DynamicCache',
    'logits_to_keep':1,'seed':config['seed'],'model_load_seconds':time.perf_counter()-started,
    'parameter_count':sum(v.numel() for v in model.parameters()),'parameter_bytes':sum(v.numel()*v.element_size() for v in model.parameters()),
    'model_config':model.config.to_dict(),'timing_protocol':'GPU events span forward+greedy argmax. Per-step host wall starts after mask/event preparation and ends after sync; whole-sequence wall includes mask/event preparation. Resident model/input; tokenization and loading excluded. Fixed output length including EOS.',
    'profiler_latency_is_performance':False}
(run/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
print('Pretrained model loaded on '+prop.name,flush=True)

def write(name,data):
    (run/name).write_text(json.dumps(data,indent=2)+'\n')

def oom_record(batch,length,stage,error):
    # Evidence for a case that does not fit device memory; never timed or estimated.
    free,total=torch.cuda.mem_get_info()
    return {'batch':batch,'prompt_length':length,'stage':stage,'error_type':type(error).__name__,
        'error':str(error).splitlines()[0],'device_total_bytes':total,'device_free_bytes':free,
        'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Case exceeded device memory under the unchanged protocol; recorded, not timed or extrapolated.'}

def inputs(batch,length):
    suffix=tokenizer.encode('\nQuestion: Explain why a larger batch can improve GPU inference throughput.\n<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    prefix=tokenizer.encode('<|im_start|>system\nYou are a helpful computer systems tutor.<|im_end|>\n<|im_start|>user\n',add_special_tokens=False)
    result=[]
    for i in range(batch):
        body=tokenizer.encode(f'Request {i}. A transformer reuses its weights across tokens. GPU memory bandwidth, matrix multiplication and the key-value cache influence inference latency. ',add_special_tokens=False)
        n=length-len(prefix)-len(suffix);assert n>0
        result.append(prefix+(body*((n+len(body)-1)//len(body)))[:n]+suffix)
    data={'batch':batch,'prompt_length':length,'input_ids':result,'text_first':tokenizer.decode(result[0])}
    data['input_sha256']=hashlib.sha256(json.dumps(result,separators=(',',':')).encode()).hexdigest()
    write(f'inputs_b{batch}_s{length}.json',data)
    return torch.tensor(result,dtype=torch.long,device='cuda'),data['input_sha256']

@torch.inference_mode()
def cache_validation(ids,tolerance=None):
    batch,length=ids.shape
    out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1)
    token=out.logits[:,-1].argmax(-1,keepdim=True)
    cached=model(input_ids=token,past_key_values=out.past_key_values,
        attention_mask=torch.ones((batch,length+1),dtype=torch.long,device='cuda'),use_cache=True,logits_to_keep=1).logits.float()
    full_ids=torch.cat([ids,token],dim=1)
    full=model(input_ids=full_ids,attention_mask=torch.ones_like(full_ids),use_cache=False,logits_to_keep=1).logits.float()
    assert torch.isfinite(full).all().item() and torch.isfinite(cached).all().item()
    error=(((cached-full).square().mean().sqrt())/full.square().mean().sqrt().clamp_min(1e-8)).item()
    result={'batch':batch,'prompt_length':length,'dtype':str(next(model.parameters()).dtype),
        'relative_logits_rms':error,'tolerance':config['cache_validation_rms_tolerance'] if tolerance is None else tolerance,
        'max_absolute_logits_difference':(cached-full).abs().max().item(),
        'top1_agreement':(cached.argmax(-1)==full.argmax(-1)).float().mean().item(),
        'cache_length_after_step':out.past_key_values.get_seq_length(),
        'scope':'Cached one-step logits versus full-prefix recomputation at the recorded dtype; not an accuracy/quantization quality benchmark.'}
    result['valid']=error<=result['tolerance']
    assert result['valid'] and result['cache_length_after_step']==length+1,result
    return result

phase='idle';capture=False;module_records=[];instrumented=False
def instrument():
    global instrumented
    instrumented=True
    # All layer/submodule labels support trace attribution; NCU filters one actual middle layer.
    for i,layer in enumerate(model.model.layers):
        modules=[('layer',layer),('attention',layer.self_attn),('mlp',layer.mlp),
            ('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]
        for name in ['q_proj','k_proj','v_proj','o_proj']:modules.append((name,getattr(layer.self_attn,name)))
        for name in ['gate_proj','up_proj','down_proj']:modules.append((name,getattr(layer.mlp,name)))
        for name,module in modules:
            original=module.forward
            def wrapped(*args,_original=original,_name=name,_layer=i,_module=module,**kwargs):
                label=f'{phase}_layer{_layer}' if _name=='layer' else _name
                with torch.profiler.record_function(label):
                    torch.cuda.nvtx.range_push(label)
                    try:
                        if capture and _layer==a.layer and hasattr(_module,'in_features'):
                            x=args[0] if args else kwargs['hidden_states']
                            module_records.append({'phase':phase,'layer':_layer,'module':_name,
                                'input_shape':list(x.shape),'in_features':_module.in_features,'out_features':_module.out_features,
                                'nominal_flops':2*math.prod(x.shape[:-1])*_module.in_features*_module.out_features})
                        return _original(*args,**kwargs)
                    finally:torch.cuda.nvtx.range_pop()
            module.forward=wrapped

@torch.inference_mode()
def sequence(ids,tokens,timed=False):
    global phase
    batch,length=ids.shape;cache=None;next_ids=ids;generated=[];timings=[];walls=[]
    torch.cuda.synchronize();wall_start=time.perf_counter()
    for step in range(tokens):
        phase='prefill' if step==0 else 'decode'
        mask=torch.ones((batch,length+step),dtype=torch.long,device='cuda')
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record()
        with torch.profiler.record_function(f'{phase}_model') if instrumented else contextlib.nullcontext():
            out=model(input_ids=next_ids,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values;next_ids=out.logits[:,-1].argmax(-1,keepdim=True)
        end.record();end.synchronize()
        timings.append(start.elapsed_time(end));walls.append((time.perf_counter()-wall)*1000)
        generated.append(next_ids)
        assert cache.get_seq_length()==length+step
    elapsed=(time.perf_counter()-wall_start)*1000
    output=torch.cat(generated,dim=1).cpu().tolist()
    return {'gpu_prefill_ms':timings[0],'gpu_decode_ms':timings[1:],'wall_prefill_ms':walls[0],
        'wall_decode_ms':walls[1:],'wall_sequence_ms':elapsed,'generated_ids':output,
        'generated_sha256':hashlib.sha256(json.dumps(output,separators=(',',':')).encode()).hexdigest(),
        'final_cache_length':cache.get_seq_length(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()}

if a.mode in ['pilot','performance']:
    grid=[(1,128)] if a.mode=='pilot' else [(b,s) for b in config['batches'] for s in config['prompt_lengths']]
    tokens=4 if a.mode=='pilot' else config['generated_tokens'];repeats=3 if a.mode=='pilot' else config['repeats']
    oom_cases=[]
    for batch,length in grid:
      stage='validation';case_rows=[]
      try:
        torch.cuda.reset_peak_memory_stats()
        ids,stamp=inputs(batch,length);validation=cache_validation(ids)
        write(f'validation_b{batch}_s{length}.json',validation)
        if a.mode=='pilot':
            # BF16 cached/full shapes round differently. Independently verify the
            # same cache path in FP32 with TF32 disabled, outside all timings.
            original_buffers={name:v.clone() for name,v in model.named_buffers()}
            try:
                model.float()
                write('fp32_cache_control.json',cache_validation(ids,tolerance=1e-4))
            finally:
                model.bfloat16()
                # Preserve original FP32 RoPE buffers as well as exact BF16 weights.
                for name,v in original_buffers.items():
                    parent,_,attribute=name.rpartition('.')
                    setattr(model.get_submodule(parent),attribute,v)
            torch.cuda.empty_cache()
        stage='warmup'
        for _ in range(config['warmup_sequences']):sequence(ids,tokens)
        stage='timed'
        for repeat in range(repeats):
            torch.cuda.reset_peak_memory_stats();result=sequence(ids,tokens,True)
            data={'batch':batch,'prompt_length':length,'generated_tokens':tokens,'repeat':repeat,'input_sha256':stamp,**result}
            case_rows.append(data)
            print(json.dumps({k:data[k] for k in ['batch','prompt_length','repeat','gpu_prefill_ms','wall_sequence_ms']}),flush=True)
      except torch.cuda.OutOfMemoryError as ex:
        record=oom_record(batch,length,stage,ex);ex=None
      else:
        record=None
        # A case's repeats are written only once all of them completed.
        with (run/'timings.jsonl').open('a') as f:f.write(''.join(json.dumps(x)+'\n' for x in case_rows))
        write(f'output_b{batch}_s{length}.json',{'first_sequence':tokenizer.decode(result['generated_ids'][0]),'generated_ids':result['generated_ids']})
      ids=validation=result=case_rows=None;gc.collect();torch.cuda.empty_cache()
      if record:
        write(f'oom_b{batch}_s{length}.json',record);oom_cases.append([batch,length])
        print(json.dumps({'out_of_memory':record}),flush=True)
    write('status.json',{'mode':a.mode,'cases':len(grid)-len(oom_cases),'oom_cases':oom_cases,'grid_cases':len(grid),
        'repeats':repeats,'generated_tokens':tokens,'complete':True})
elif a.mode=='trace':
  try:
    ids,stamp=inputs(a.batch,a.length)
    sequence(ids,4);sequence(ids,4);instrument()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,with_flops=True) as prof:result=sequence(ids,4)
    import gzip,shutil
    local=pathlib.Path(os.environ.get('APP_LOCAL_DIR',str(run)));local.mkdir(parents=True,exist_ok=True)
    temporary=local/f'trace_b{a.batch}_s{a.length}.json'
    prof.export_chrome_trace(str(temporary))
    with temporary.open('rb') as src,gzip.open(run/'trace.json.gz','wb') as dst:shutil.copyfileobj(src,dst)
    write('trace_case.json',{'batch':a.batch,'prompt_length':a.length,'generated_tokens':4,'input_sha256':stamp,
        'scope':'One full-model prefill plus three decode calls; trace is instrumented, not performance timing.',
        'result':result})
  except torch.cuda.OutOfMemoryError as ex:
    write('oom.json',oom_record(a.batch,a.length,'trace',ex));sys.exit(3)
else:
  try:
    ids,stamp=inputs(a.batch,a.length);validation=cache_validation(ids);write('validation.json',validation)
    sequence(ids,4);sequence(ids,4);instrument()
    @torch.inference_mode()
    def counter_call():
        global phase,capture
        cache=None;next_ids=ids
        if a.phase=='decode':
            for step in range(5):
                phase='warmup'
                out=model(input_ids=next_ids,attention_mask=torch.ones((a.batch,a.length+step),dtype=torch.long,device='cuda'),
                    past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache=out.past_key_values;next_ids=out.logits[:,-1].argmax(-1,keepdim=True)
        phase=a.phase;capture=True;torch.cuda.synchronize()
        assert int(torch.cuda.cudart().cudaProfilerStart())==0
        out=model(input_ids=next_ids,attention_mask=torch.ones((a.batch,a.length+(5 if cache is not None else 0)),dtype=torch.long,device='cuda'),
            past_key_values=cache,use_cache=True,logits_to_keep=1)
        torch.cuda.synchronize();assert int(torch.cuda.cudart().cudaProfilerStop())==0
        capture=False
        assert torch.isfinite(out.logits).all().item()
        return {'batch':a.batch,'prompt_length':a.length,'phase':a.phase,'layer':a.layer,'input_sha256':stamp,
            'input_query_tokens':next_ids.shape[1],'attention_kv_tokens':out.past_key_values.get_seq_length(),
            'linear_operators':module_records,'finite_logits':True,'validation':validation,
            'scope':'Selected middle layer executed inside full pretrained model; NCU filters its NVTX range. Nominal linear FLOPs; profiler time excluded from inference performance.'}
    write('counter_case.json',counter_call())
  except torch.cuda.OutOfMemoryError as ex:
    write('oom.json',oom_record(a.batch,a.length,'counters '+a.phase,ex));sys.exit(3)
(run/'PYTHON_COMPLETE').write_text('Application protocol completed.\n')
