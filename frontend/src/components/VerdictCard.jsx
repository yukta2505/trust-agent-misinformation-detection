export default function VerdictCard({ result }) {
  const isOOC = result.verdict === "OUT-OF-CONTEXT"
  const color = isOOC ? "var(--ooc)" : "var(--pristine)"

  return (
    <div style={{ marginTop: "1.5rem", background: "var(--surface)",
                  borderRadius: 12, padding: "1.5rem",
                  border: `2px solid ${color}` }}>
      <div style={{ display: "flex", alignItems: "center",
                    gap: "1rem", marginBottom: "1rem" }}>
        <span style={{ fontSize: "2rem" }}>{isOOC ? "⚠" : "✓"}</span>
        <div>
          <div style={{ color, fontWeight: 800, fontSize: "1.4rem" }}>
            {result.verdict.replace("-", " ")}
          </div>
          <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
            {result.confidence_percent}% confidence
            · processed in {result.processing_time_sec}s
          </div>
        </div>
        {/* Confidence bar */}
        <div style={{ flex: 1, height: 8, background: "var(--border)",
                      borderRadius: 4, overflow: "hidden", marginLeft: "auto" }}>
          <div style={{ width: `${result.confidence_percent}%`,
                        height: "100%", background: color,
                        transition: "width 1s ease" }} />
        </div>
      </div>

      <p style={{ lineHeight: 1.7, marginBottom: "1rem" }}>
        {result.explanation}
      </p>

      {result.caption && (
        <div style={{ color: "var(--muted)", fontSize: "0.85rem",
                      fontFamily: "'DM Mono', monospace",
                      background: "#0d0f14", padding: "0.5rem 0.75rem",
                      borderRadius: 6 }}>
          Caption: {result.caption}
        </div>
      )}

      {result.flags.length > 0 && (
        <div style={{ marginTop: "1rem" }}>
          <div style={{ color: "var(--ooc)", fontWeight: 700,
                        fontSize: "0.85rem", marginBottom: "0.4rem" }}>
            RED FLAGS
          </div>
          {result.flags.map((f, i) => (
            <div key={i} style={{ color: "#fca5a5", fontSize: "0.875rem",
                                   padding: "0.2rem 0" }}>⚠ {f}</div>
          ))}
        </div>
      )}

      {result.key_evidence_for_verdict.length > 0 && (
        <div style={{ marginTop: "1rem" }}>
          <div style={{ color: color, fontWeight: 700,
                        fontSize: "0.85rem", marginBottom: "0.4rem" }}>
            KEY EVIDENCE
          </div>
          {result.key_evidence_for_verdict.map((e, i) => (
            <div key={i} style={{ fontSize: "0.875rem", padding: "0.2rem 0",
                                   color: "var(--text)" }}>• {e}</div>
          ))}
        </div>
      )}
    </div>
  )
}