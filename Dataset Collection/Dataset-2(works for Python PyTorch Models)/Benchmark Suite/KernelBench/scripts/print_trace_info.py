import torch, sys
from pathlib import Path

MODEL_PATH = Path('KernelBench/level3/6_GoogleNetInceptionModule.py')
if not MODEL_PATH.exists():
    print('Model file not found:', MODEL_PATH)
    sys.exit(1)

src = MODEL_PATH.read_text()
ctx = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
exec(src, ctx)
Model = ctx.get('Model')
get_init_inputs = ctx.get('get_init_inputs')
get_inputs = ctx.get('get_inputs')

print('Model found:', Model is not None)
print('get_init_inputs found:', get_init_inputs is not None)
print('get_inputs found:', get_inputs is not None)

init_args = get_init_inputs()
print('get_init_inputs ->', init_args)
inputs = get_inputs()
print('Input count:', len(inputs))
for i, inp in enumerate(inputs):
    if isinstance(inp, torch.Tensor):
        print(f' Input[{i}] shape:', tuple(inp.shape), 'dtype:', inp.dtype)
    else:
        print(f' Input[{i}] type:', type(inp))

# Instantiate model
try:
    model = Model(*init_args)
    print('Model instantiated')
except Exception as e:
    print('Model init failed:', e)
    sys.exit(1)

# Print conv module weights
print('\nConv modules and weight shapes:')
for name, m in model.named_modules():
    if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        try:
            w = m.weight
            print(f'  {name}: weight.shape =', tuple(w.shape), 'groups=', getattr(m, 'groups', 1))
        except Exception as e:
            print('  Error reading weight for', name, e)

# Prepare trace inputs (zeros with same shapes)
trace_inputs = []
for inp in inputs:
    if isinstance(inp, torch.Tensor):
        trace_inputs.append(torch.zeros(inp.shape, dtype=inp.dtype))
    else:
        trace_inputs.append(inp)

# Attempt TorchScript tracing
print('\nAttempting TorchScript trace...')
try:
    if len(trace_inputs) == 1:
        traced = torch.jit.trace(model, trace_inputs[0], check_trace=False)
    else:
        traced = torch.jit.trace(model, tuple(trace_inputs), check_trace=False)
    print('Tracing succeeded. Graph node kinds:')
    for node in traced.graph.nodes():
        print(' -', node.kind())
except Exception as e:
    print('Tracing failed:', e)
    print('\nAttempting torch.fx symbolic_trace as fallback...')
    try:
        import torch.fx as fx
        traced_fx = fx.symbolic_trace(model)
        print('FX graph nodes:')
        for n in traced_fx.graph.nodes:
            print(' -', n.op, n.target)
    except Exception as e2:
        print('FX trace failed:', e2)

print('\nDone')
