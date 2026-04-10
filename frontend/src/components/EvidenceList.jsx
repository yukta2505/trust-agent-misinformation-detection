export default function EvidenceList({ evidence }) {
  if (!evidence || evidence.length === 0) return null

  return (
    <div style={{ marginTop: "1rem", background: "var(--surface)",
                  borderRadius: 12, padding: "1.5rem",
                  border: "1px solid var(--border)" }}>
      <div style={{ fontWeight: 700, marginBottom: "1rem",
                    fontSize: "0.85rem", color: "var(--muted)" }}>
        RETRIEVED EVIDENCE ({evidence.length})
      </div>
      {evidence.map((item, i) => (
        <div key={i} style={{ padding: "0.75rem",
                               background: "#0d0f14", borderRadius: 8,
                               marginBottom: "0.5rem" }}>
          <div style={{ fontWeight: 600, fontSize: "0.9rem",
                        marginBottom: "0.25rem" }}>
            {item.title || "(no title)"}
          </div>
          {item.snippet && (
            <div style={{ color: "var(--muted)", fontSize: "0.8rem",
                           lineHeight: 1.5, marginBottom: "0.25rem" }}>
              {item.snippet}
            </div>
          )}
          <div style={{ display: "flex", gap: "1rem",
                        fontSize: "0.75rem", color: "var(--muted)" }}>
            <span>{item.source}</span>
            <span>score: {item.score?.toFixed(3) ?? "—"}</span>
            {item.url && (
              <a href={item.url} target="_blank" rel="noreferrer"
                 style={{ color: "var(--accent)" }}>
                View source →
              </a>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}