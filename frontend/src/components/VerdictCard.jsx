function clampPct(value) {
  const num = Number(value)
  if (!Number.isFinite(num)) return 0
  return Math.max(0, Math.min(100, Math.round(num)))
}

export default function VerdictCard({ result }) {
  const isOOC = result?.verdict === "OUT-OF-CONTEXT"
  const color = isOOC ? "var(--ooc)" : "var(--pristine)"
  const pct = clampPct(result?.confidence_percent)

  return (
    <section
      className="panel panel--pad"
      style={{
        borderColor: isOOC ? "rgba(239,68,68,.30)" : "rgba(34,197,94,.25)",
        boxShadow: isOOC ? "var(--shadow), var(--ringDanger)" : "var(--shadow), var(--ringGood)",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14, flexWrap: "wrap" }}>
        <div
          aria-hidden
          style={{
            width: 40,
            height: 40,
            borderRadius: 14,
            display: "grid",
            placeItems: "center",
            background: "rgba(255,255,255,.04)",
            border: "1px solid rgba(255,255,255,.10)",
          }}
        >
          <span style={{ fontSize: 18, color }}>{isOOC ? "⚠" : "✓"}</span>
        </div>

        <div style={{ flex: "1 1 280px", minWidth: 260 }}>
          <div className="panel__kicker">Verdict</div>
          <div className="panel__title" style={{ color }}>
            {String(result?.verdict || "").replaceAll("-", " ")}
          </div>

          <div className="kv">
            <span>
              <strong className="badge__mono" style={{ color }}>{pct}%</strong> <span>confidence</span>
            </span>
            <span>•</span>
            <span>
              processed in <strong className="badge__mono">{result?.processing_time_sec}s</strong>
            </span>
          </div>

          <div style={{ marginTop: 12 }}>
            <div className="meter" aria-label={`Confidence ${pct}%`}>
              <div className="meter__fill" style={{ "--w": `${pct}%`, "--c": color }} />
            </div>
          </div>
        </div>

        <span className="badge" style={{ borderColor: "rgba(255,255,255,.14)" }}>
          <span style={{ width: 8, height: 8, borderRadius: 999, background: color }} />
          <span className="badge__mono">{pct}%</span>
        </span>
      </div>

      <div className="divider" />

      <p style={{ margin: 0, color: "var(--text)", lineHeight: 1.75, fontSize: 16 }}>
        {result?.explanation}
      </p>

      {result?.caption && (
        <div style={{ marginTop: 14 }}>
          <div className="panel__kicker">Caption</div>
          <div
            className="badge badge__mono"
            style={{
              marginTop: 8,
              display: "block",
              borderRadius: 14,
              padding: "10px 12px",
              background: "rgba(0,0,0,.18)",
              borderColor: "rgba(255,255,255,.10)",
              color: "var(--muted)",
            }}
          >
            {result.caption}
          </div>
        </div>
      )}

      {Array.isArray(result?.flags) && result.flags.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="panel__kicker" style={{ color: "rgba(239,68,68,.9)" }}>Red Flags</div>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, color: "rgba(252,165,165,.95)", lineHeight: 1.7 }}>
            {result.flags.map((f, i) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      )}

      {Array.isArray(result?.key_evidence_for_verdict) && result.key_evidence_for_verdict.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="panel__kicker" style={{ color }}>Key Evidence</div>
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, color: "var(--text)", lineHeight: 1.7 }}>
            {result.key_evidence_for_verdict.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </div>
      )}
    </section>
  )
}
