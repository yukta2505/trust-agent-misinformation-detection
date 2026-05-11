function clamp01(v) {
  const n = Number(v)
  if (!Number.isFinite(n)) return 0
  return Math.max(0, Math.min(1, n))
}

function ScoreRow({ label, score01, color }) {
  const pct = Math.round(clamp01(score01) * 100)
  return (
    <div style={{ padding: "10px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
        <div style={{ fontWeight: 700, color: "var(--text)" }}>{label}</div>
        <span className="badge badge__mono" style={{ padding: "4px 10px", borderColor: "rgba(255,255,255,.12)" }}>
          {pct}%
        </span>
      </div>
      <div style={{ marginTop: 10 }} className="meter">
        <div className="meter__fill" style={{ "--w": `${pct}%`, "--c": color }} />
      </div>
    </div>
  )
}

export default function AgentScores({ result }) {
  return (
    <section className="panel panel--pad">
      <div className="panel__kicker">Agent Scores</div>
      <div className="panel__title">Signals breakdown</div>

      <div className="divider" />

      <ScoreRow label="Entity Consistency (35%)" score01={result?.entity_score} color="#7c83ff" />
      <ScoreRow label="Temporal Consistency (35%)" score01={result?.temporal_score} color="var(--amber)" />
      <ScoreRow label="Source Credibility (30%)" score01={result?.credibility_score} color="var(--pristine)" />
      <ScoreRow label="Claim Plausibility (soft)" score01={result?.plausibility_score ?? 0.7} color="#f97316" />

      <div className="divider" />

      <ScoreRow label="Final Weighted Score" score01={result?.final_score} color="var(--accent)" />
    </section>
  )
}
