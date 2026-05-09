export default function LoadingSpinner() {
  return (
    <div style={{ textAlign: "center", padding: "3rem",
                  color: "var(--muted)" }}>
      <div style={{ width: 40, height: 40, border: "3px solid var(--border)",
                    borderTopColor: "var(--accent)", borderRadius: "50%",
                    animation: "spin 0.8s linear infinite",
                    margin: "0 auto 1rem" }} />
      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      <div>Running multi-agent analysis...</div>
      <div style={{ fontSize: "0.8rem", marginTop: "0.4rem" }}>
        Entity · Temporal · Credibility agents running in parallel
      </div>
    </div>
  )
}