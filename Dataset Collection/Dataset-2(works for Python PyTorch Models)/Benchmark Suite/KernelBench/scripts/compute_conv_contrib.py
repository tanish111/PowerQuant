import torch
from pathlib import Path

MODEL_PATH = Path('KernelBench/level3/6_GoogleNetInceptionModule.py')
src = MODEL_PATH.read_text()
ctx = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
exec(src, ctx)
Model = ctx.get('Model')
get_init_inputs = ctx.get('get_init_inputs')
get_inputs = ctx.get('get_inputs')

init_args = get_init_inputs()
inputs = get_inputs()

# scaling logic matching extractor
total_numel = sum(inp.numel() for inp in inputs if isinstance(inp, torch.Tensor))
max_numel = 10_000_000
if total_numel <= max_numel:
    trace_inputs = [torch.zeros(inp.shape, dtype=inp.dtype) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
    scale_factor = 1.0
else:
    scale = (max_numel / total_numel) ** 0.5
    scale = max(0.01, scale)
    trace_inputs = []
    scaled_numel = 0
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            new_shape = tuple(max(2, int(s * scale)) for s in inp.shape)
            trace_inputs.append(torch.zeros(new_shape, dtype=inp.dtype))
            scaled_numel += int(torch.zeros(new_shape).numel())
        else:
            trace_inputs.append(inp)
    scale_factor = total_numel / scaled_numel if scaled_numel > 0 else 1.0

# collect conv outputs via hooks
model = Model(*init_args)
module_output_shapes = {}
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        def make_hook(n):
            def hook(mod, inp, out):
                module_output_shapes[n] = tuple(out.shape)
            return hook
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

model.eval()
with torch.no_grad():
    if len(trace_inputs) == 1:
        _ = model(trace_inputs[0])
    else:
        _ = model(*tuple(trace_inputs))

for h in hooks:
    h.remove()

# compute conv flops using module params and captured shapes (same formula as extractor)
conv_details = []
total_conv_flops = 0
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        w = getattr(module, 'weight', None)
        out_shape = module_output_shapes.get(name)
        if w is None or out_shape is None:
            continue
        w_shape = tuple(w.shape)
        groups = getattr(module, 'groups', 1)
        flops = 0
        try:
            if len(out_shape) == 4 and len(w_shape) >= 4:
                n, cout, h_out, w_out = out_shape
                kH, kW = w_shape[-2], w_shape[-1]
                cin = w_shape[1] * groups if len(w_shape) > 1 else 1
                flops = int(2 * n * cout * h_out * w_out * (cin // groups) * kH * kW)
            elif len(out_shape) == 3 and len(w_shape) >= 3:
                n, cout, l_out = out_shape
                k = w_shape[-1]
                cin = w_shape[1] * groups if len(w_shape) > 1 else 1
                flops = int(2 * n * cout * l_out * (cin // groups) * k)
            elif len(out_shape) == 5 and len(w_shape) >= 5:
                n, cout, d_out, h_out, w_out = out_shape
                kD, kH, kW = w_shape[-3], w_shape[-2], w_shape[-1]
                cin = w_shape[1] * groups if len(w_shape) > 1 else 1
                flops = int(2 * n * cout * d_out * h_out * w_out * (cin // groups) * kD * kH * kW)
        except Exception:
            flops = 0
        conv_details.append((name, w_shape, out_shape, groups, flops))
        total_conv_flops += flops

# print results
print('Scaling:')
print('  total_numel=', total_numel)
if scale_factor != 1.0:
    print('  applied scale factor to trace inputs, scale_factor =', round(scale_factor,6))
else:
    print('  no scaling applied')

print('\nPer-conv FLOP contributions (using scaled trace shapes):')
for name, w_shape, out_shape, groups, flops in conv_details:
    print(f" - {name}: weight={w_shape}, out={out_shape}, groups={groups}, flops={flops} (~{flops/1e6:.3f} MFLOPs)")
print(f'\nTotal conv FLOPs = {total_conv_flops} (~{total_conv_flops/1e6:.3f} MFLOPs)')

# compare to model_features.csv total if present
mf = Path('model_features.csv')
if mf.exists():
    import csv
    with mf.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row['name'].startswith('6_GoogleNetInceptionModule'):
                reported = float(row['total_flops_m']) * 1e6
                print(f"\nReported total_flops in CSV = {reported} ({reported/1e6:.3f} MFLOPs)")
                print(f"Conv fraction = {total_conv_flops/reported:.3f}")
                break
