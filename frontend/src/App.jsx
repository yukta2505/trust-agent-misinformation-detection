import { useState } from "react"
import axios from "axios"
import UploadForm from "./components/UploadForm"
import VerdictCard from "./components/VerdictCard"
import AgentScores from "./components/AgentScores"
import EvidenceList from "./components/EvidenceList"
import LoadingSpinner from "./components/LoadingSpinner"

const API = "http://localhost:8000"

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(image, claim) {
    setLoading(true); setError(null); setResult(null)
    const form = new FormData()
    form.append("image", image)
    form.append("claim", claim)
    try {
      const { data } = await axios.post(`${API}/analyse`, form)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail || "Server error")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1rem" }}>
      <h1 style={{ fontWeight: 800, fontSize: "2rem", marginBottom: "0.25rem" }}>
        TRUST-AGENT
      </h1>
      <p style={{ color: "var(--muted)", marginBottom: "2rem" }}>
        Out-of-context misinformation detection
      </p>

      <UploadForm onSubmit={handleSubmit} loading={loading} />

      {loading && <LoadingSpinner />}
      {error && (
        <div style={{ color: "var(--ooc)", padding: "1rem",
                      background: "#1a0a0a", borderRadius: 8, marginTop: "1rem" }}>
          Error: {error}
        </div>
      )}
      {result && (
        <>
          <VerdictCard result={result} />
          <AgentScores result={result} />
          <EvidenceList evidence={result.evidence} />
        </>
      )}
    </div>
  )
}
