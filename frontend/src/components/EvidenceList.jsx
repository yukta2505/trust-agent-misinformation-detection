function hostFromUrl(url) {
  try {
    return new URL(url).host.replace(/^www\./, "")
  } catch {
    return null
  }
}

export default function EvidenceList({ evidence }) {
  if (!Array.isArray(evidence) || evidence.length === 0) return null

  return (
    <section className="panel panel--pad">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div className="panel__kicker">News Retrieval</div>
          <div className="panel__title">Retrieved evidence ({evidence.length})</div>
        </div>
        <span className="badge">
          <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--accent)" }} />
          <span className="badge__mono">{evidence.length} items</span>
        </span>
      </div>

      <div className="divider" />

      <div style={{ display: "grid", gap: 10 }}>
        {evidence.map((item, i) => {
          const host = item?.url ? hostFromUrl(item.url) : null
          const score = Number(item?.score)
          const scoreText = Number.isFinite(score) ? score.toFixed(3) : "—"

          return (
            <details
              key={i}
              style={{
                borderRadius: 14,
                border: "1px solid rgba(255,255,255,.10)",
                background: "rgba(0,0,0,.18)",
                overflow: "hidden",
              }}
              open={i < 1}
            >
              <summary
                style={{
                  listStyle: "none",
                  cursor: "pointer",
                  padding: "12px 12px",
                  display: "flex",
                  gap: 12,
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                }}
              >
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 900, fontSize: 15, color: "var(--text)" }}>
                    {item?.title || "(no title)"}
                  </div>

                  <div style={{ marginTop: 8, display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                    <span className="badge badge__mono" style={{ padding: "4px 10px" }}>
                      score {scoreText}
                    </span>
                    {(item?.source || host) && (
                      <span className="badge" style={{ padding: "4px 10px", color: "var(--muted)" }}>
                        {item?.source || host}
                      </span>
                    )}
                    {host && item?.source && item.source !== host && (
                      <span className="badge" style={{ padding: "4px 10px", color: "var(--muted)" }}>
                        {host}
                      </span>
                    )}
                  </div>
                </div>

                {item?.url && (
                  <a
                    href={item.url}
                    target="_blank"
                    rel="noreferrer"
                    className="badge"
                    style={{
                      textDecoration: "none",
                      borderColor: "rgba(124,131,255,.30)",
                      background: "rgba(124,131,255,.10)",
                      whiteSpace: "nowrap",
                    }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <span className="badge__mono">Open</span> →
                  </a>
                )}
              </summary>

              {(item?.snippet || item?.url) && (
                <div style={{ padding: "0 12px 12px" }}>
                  {item?.snippet && (
                    <p style={{ margin: 0, color: "var(--muted)", lineHeight: 1.7, fontSize: 14 }}>
                      {item.snippet}
                    </p>
                  )}
                </div>
              )}
            </details>
          )
        })}
      </div>
    </section>
  )
}
