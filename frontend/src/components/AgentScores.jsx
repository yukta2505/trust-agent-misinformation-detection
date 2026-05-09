function ScoreBar({ label, score, color }) {
  const pct = Math.round(score * 100)
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    marginBottom: "0.3rem", fontSize: "0.8rem" }}>
        <span style={{ color: "var(--muted)" }}>{label}</span>
        <span style={{ fontFamily: "'DM Mono', monospace",
                       color }}>{pct}%</span>
      </div>
      <div style={{ height: 6, background: "var(--border)",
                    borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%",
                      background: color, borderRadius: 3,
                      transition: "width 1s ease" }} />
      </div>
    </div>
  )
}

export default function AgentScores({ result }) {
  return (
    <div style={{ marginTop: "1rem", background: "var(--surface)",
                  borderRadius: 12, padding: "1.5rem",
                  border: "1px solid var(--border)" }}>
      <div style={{ fontWeight: 700, marginBottom: "1rem",
                    fontSize: "0.85rem", color: "var(--muted)" }}>
        AGENT SCORES
      </div>
      <ScoreBar label="Entity Consistency (35%)"
                score={result.entity_score} color="#818cf8" />
      <ScoreBar label="Temporal Consistency (35%)"
                score={result.temporal_score} color="var(--amber)" />
      <ScoreBar label="Source Credibility (30%)"
                score={result.credibility_score} color="#34d399" />
      <div style={{ marginTop: "1rem", paddingTop: "1rem",
                    borderTop: "1px solid var(--border)" }}>
        <ScoreBar label="Final Weighted Score"
                  score={result.final_score} color="var(--accent)" />
      </div>
    </div>
  )
}