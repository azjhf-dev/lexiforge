"""Fit planner: work out which quantization and layer split fits a GPU.

Given a GGUF file and a VRAM budget, reports model size, KV cache cost per
token, the maximum usable context, and - for MoE models - the savings from
keeping expert weights in host RAM (the "one 24 GB card serves a 94 GB MoE"
trick: attention and router on the card, experts fetched per token).
"""

from __future__ import annotations

from .gguf import GGUF

GIB = 1 << 30


def analyze(path, vram_gib=None, context=32768, kv_bytes_per_elem=2):
    g = GGUF(path)
    md = g.metadata
    arch = md.get("general.architecture", "llama")
    p = arch + "."
    embd = int(md[p + "embedding_length"])
    layers = int(md[p + "block_count"])
    n_kv = int(md.get(p + "attention.head_count_kv", md[p + "attention.head_count"]))
    head_dim = int(md.get(p + "attention.key_length", embd // int(md[p + "attention.head_count"])))
    n_experts = int(md.get(p + "expert_count", 0))

    total = g.total_bytes()
    expert_bytes = 0
    attn_bytes = 0
    for t in g.tensors.values():
        if "_exps." in t.name:
            expert_bytes += t.nbytes
        elif "attn" in t.name or "norm" in t.name or "embd" in t.name or "output" in t.name:
            attn_bytes += t.nbytes

    kv_per_token = layers * n_kv * head_dim * 2 * kv_bytes_per_elem  # k and v
    kv_cache = kv_per_token * context

    report = {
        "arch": arch, "layers": layers, "embd": embd, "experts": n_experts,
        "model_gib": round(total / GIB, 2),
        "expert_gib": round(expert_bytes / GIB, 2),
        "non_expert_gib": round((total - expert_bytes) / GIB, 2),
        "kv_gib_at_ctx": round(kv_cache / GIB, 2),
        "context": context,
    }
    if vram_gib:
        budget = vram_gib * GIB
        report["fits_full"] = total + kv_cache <= budget
        if n_experts:
            resident = total - expert_bytes + kv_cache
            report["fits_with_expert_offload"] = resident <= budget
            report["resident_gib_with_offload"] = round(resident / GIB, 2)
        if not report["fits_full"]:
            # how much context fits if weights stay resident
            room = budget - total
            report["max_context_full_offload_free"] = max(0, int(room // kv_per_token))
    return report


def fmt(report) -> str:
    lines = [
        f"arch {report['arch']}, {report['layers']} layers, embd {report['embd']}",
        f"weights {report['model_gib']} GiB "
        f"({report['non_expert_gib']} GiB non-expert, {report['expert_gib']} GiB experts)",
        f"KV cache at {report['context']} ctx: {report['kv_gib_at_ctx']} GiB",
    ]
    if "fits_full" in report:
        lines.append(f"fits fully in VRAM: {report['fits_full']}")
        if report.get("fits_with_expert_offload") is not None:
            lines.append(
                f"fits with expert offload to RAM: {report['fits_with_expert_offload']} "
                f"(resident {report.get('resident_gib_with_offload')} GiB)")
        if not report["fits_full"] and "max_context_full_offload_free" in report:
            lines.append(
                f"max context with weights resident: "
                f"{report['max_context_full_offload_free']} tokens")
    return "\n".join(lines)
