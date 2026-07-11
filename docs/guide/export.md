# Exporting to ONNX

`echelon3 export` turns a trained checkpoint into a single self-contained ONNX
graph. The key idea: the exporter fuses your **preprocess**, the **network**, and
an optional **postprocess** into one model, so the exported graph accepts a raw
image tensor and returns final outputs — no Python preprocessing needed at
inference time.

!!! note "Install the export extra"
    ONNX export needs `onnx` and `onnxruntime`:

    ```bash
    pip install echelon3[export]
    ```

## The `export` section

`create_exporters` reads three things from the `export` section:

- **`preprocess`** — an ordered map of `torch.nn.Module`s (identical shape to
  `transform.*.preprocess`). Defaults to `Identity` if omitted.
- **`postprocess`** — the same, applied to the network's output. Defaults to
  `Identity`.
- **`exporters`** — a map of named exporter triples. Each one is built with the
  network plus the fused preprocess/postprocess and then run.

The exporter base class wraps everything into one module whose forward pass is:

```
preprocess  →  net  →  postprocess
```

That fused module is what gets written to ONNX. Put pixel scaling / normalization
in `preprocess` and any decode step (softmax-to-score, argmax-to-mask,
sigmoid-threshold) in `postprocess`, and the graph becomes drop-in for a runtime
that feeds it raw images.

## `OnnxExporter` keys

`module: echelon3.exporters.onnx` / `type: OnnxExporter`:

| Key | Meaning |
| --- | --- |
| `target` | Output `.onnx` path (parent directories are created). |
| `input_names` | Names for the graph inputs. |
| `output_names` | Names for the graph outputs. |
| `input_shape` | Shape of the example input used to trace/script (e.g. `[1, 3, 64, 64]`). |
| `opset` | ONNX opset version (default `18`). |
| `use_tracing` | `true` traces the module; `false` runs `torch.jit.script` first (default `false`). |
| `dynamic_axes` | Optional map marking axes as dynamic (e.g. a variable batch size). |
| `do_constant_folding` | Fold constants during export (default `true`). |
| `use_aten_fallback` | Export with the ATen fallback operator set (default `false`). |

!!! note "The example input is `uint8`"
    The exporter builds its dummy input as a `uint8` tensor in `[0, 255]` with
    `input_shape`. That is why preprocessing (e.g. `To01`) belongs *inside* the
    exported graph: the resulting ONNX model consumes a raw `uint8` NCHW image
    directly.

## Running the export

```bash
echelon3 export --config-dir configs --config-name my_experiment
```

The exporter builds the network from `net`, and — if a `target` section is
present — loads the latest checkpoint from `target.path` (falling back to a
`DataParallel`-style load for `module.`-prefixed keys). It then runs every
exporter under `export.exporters` in turn. With no `target`, it exports the
freshly-initialized network (useful for shape/graph checks).

## Example

The `export` section from `examples/configs/smoke.yaml` folds a `To01` step into
the graph and writes one ONNX file:

```yaml
export:
  preprocess:
    to01:
      name: to01
      module: echelon3.transforms.preprocess.basic
      type: To01
  exporters:
    onnx:
      module: echelon3.exporters.onnx
      type: OnnxExporter
      config:
        target: ${oc.env:SMOKE_TARGET,./targets/smoke}/smoke.onnx
        input_names: [images]
        output_names: [logits]
        input_shape: [1, 3, 64, 64]
        use_tracing: true
        opset: 18
```

Run it after training the smoke model:

```bash
SMOKE_TARGET=./targets/smoke \
    echelon3 export --config-dir examples/configs --config-name smoke
```

This writes `./targets/smoke/smoke.onnx`. The graph takes a `uint8` `images`
input of shape `[1, 3, 64, 64]`, scales it to `[0, 1]` internally via the fused
`To01`, runs the classifier, and returns `logits`.

!!! tip "Dynamic batch size"
    To accept any batch size at inference, mark the batch axis dynamic:

    ```yaml
    config:
      input_shape: [1, 3, 64, 64]
      dynamic_axes:
        images: { 0: batch }
        logits: { 0: batch }
    ```

## Adding a postprocess

To emit class scores instead of raw logits, append a postprocess step:

```yaml
export:
  preprocess:
    to01: { name: to01, module: echelon3.transforms.preprocess.basic, type: To01 }
  postprocess:
    score: { name: score, module: echelon3.transforms.preprocess.basic, type: ScoreFromLogits, config: { score_class: 1 } }
  exporters:
    onnx: { ... }
```

For segmentation, `echelon3.transforms.preprocess.segmentation.ToSegmentationMask`
or `ToBinarySegmentationMask` make good postprocess steps.

## Next

- [Config Schema](../reference/config-schema.md) — the `export` section in context.
- [Built-in Components](../reference/components.md) — preprocess/postprocess modules.
</content>
