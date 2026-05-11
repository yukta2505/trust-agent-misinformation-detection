import { useEffect, useState } from "react"
import { Routes, Route, NavLink } from "react-router-dom"
import axios from "axios"

import UploadForm from "./components/UploadForm"
import VerdictCard from "./components/VerdictCard"
import AgentScores from "./components/AgentScores"
import EvidenceList from "./components/EvidenceList"
import LoadingSpinner from "./components/LoadingSpinner"

const API = "http://localhost:8000"

function Shell({ children }) {
  const [theme, setTheme] = useState(() => {
    return document.documentElement.getAttribute("data-theme") || "dark"
  })

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
    localStorage.setItem("theme", theme)
  }, [theme])

  function toggleTheme() {
    setTheme((t) => (t === "dark" ? "light" : "dark"))
  }
  return (
    <div className="app">
      <header className="topbar">
        <div className="container topbar__inner">
          <NavLink to="/" className="brand">
            <span className="brand__mark" aria-hidden />
            <span className="brand__text">TRUST-AGENT</span>
          </NavLink>

          <nav className="nav">
            <NavLink to="/" end className={({ isActive }) => `nav__link ${isActive ? "is-active" : ""}`}>
              Home
            </NavLink>
            <NavLink to="/about" className={({ isActive }) => `nav__link ${isActive ? "is-active" : ""}`}>
              About
            </NavLink>
            <NavLink to="/contact" className={({ isActive }) => `nav__link ${isActive ? "is-active" : ""}`}>
              Contact
            </NavLink>
          </nav>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <nav className="nav">{/* links */}</nav>

            <button className="themeToggle" onClick={toggleTheme} type="button">
              <span className="themeToggle__dot" aria-hidden />
              <span style={{ fontFamily: "var(--font-mono)", fontSize: 13 }}>
                {theme === "dark" ? "Dark" : "Light"}
              </span>
            </button>
          </div>
        </div>
      </header>

      <main className="container main">{children}</main>

      <footer className="footer">
        <div className="container footer__inner">
          <span className="muted">© {new Date().getFullYear()} TRUST-AGENT</span>
          <span className="muted">Out-of-context misinformation detection</span>
        </div>
      </footer>
    </div>
  )
}

function HomePage() {
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
    <>
      <section className="hero">
        <div className="hero__bg" aria-hidden />
        <div className="hero__inner">
          <h1 className="h1">Detect out-of-context image misinformation.</h1>
          <p className="lead">
            Upload an image + a claim. Get a clear verdict, agent scores, and evidence/news retrieval you can audit.
          </p>
        </div>
      </section>

      <section className="grid2">
        <div className="card card--pad">
          <div className="card__title">Analyse</div>
          <div className="card__sub">Start with an image and a short claim.</div>
          <div style={{ marginTop: "1rem" }}>
            <UploadForm onSubmit={handleSubmit} loading={loading} />
          </div>

          {loading && <div style={{ marginTop: "1rem" }}><LoadingSpinner /></div>}
          {error && (
            <div className="alert alert--error" style={{ marginTop: "1rem" }}>
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        <div className="card card--pad">
          <div className="card__title">How to read results</div>
          <ul className="list">
            <li><span className="pill">Verdict</span> final decision + confidence.</li>
            <li><span className="pill">Agent scores</span> independent signals, side-by-side.</li>
            <li><span className="pill">News retrieval</span> sources/evidence used for reasoning.</li>
          </ul>
          <div className="muted" style={{ marginTop: "0.75rem" }}>
            Tip: keep claims specific (who/where/when) for better retrieval.
          </div>
        </div>
      </section>

      {result && (
        <div className="stack">
          <section className="card card--pad">
            <div className="sectionHead">
              <div>
                <div className="sectionHead__title">Verdict</div>
                <div className="sectionHead__sub">A crisp summary of what the system believes.</div>
              </div>
            </div>
            <VerdictCard result={result} />
          </section>

          <section className="card card--pad">
            <div className="sectionHead">
              <div>
                <div className="sectionHead__title">Agent Scores</div>
                <div className="sectionHead__sub">Separate section for interpretability.</div>
              </div>
            </div>
            <AgentScores result={result} />
          </section>

          <section className="card card--pad">
            <div className="sectionHead">
              <div>
                <div className="sectionHead__title">News Retrieval / Evidence</div>
                <div className="sectionHead__sub">What the system pulled in to support the decision.</div>
              </div>
            </div>
            <EvidenceList evidence={result.evidence} />
          </section>
        </div>
      )}
    </>
  )
}

function AboutPage() {
  return (
    <section className="card card--pad">
      <div className="sectionHead">
        <div>
          <div className="sectionHead__title">About</div>
          <div className="sectionHead__sub">What TRUST-AGENT does and how to use it well.</div>
        </div>
      </div>

      <div className="prose">
        <p>
          TRUST-AGENT helps detect out-of-context misinformation by combining retrieval (news/evidence)
          with multiple agent signals and a final verdict.
        </p>
        <h3>Best practices</h3>
        <ul>
          <li>Use short, checkable claims (who/where/when).</li>
          <li>Verify evidence links and timestamps.</li>
          <li>Treat scores as signals, not absolute truth.</li>
        </ul>
        <h3>Privacy</h3>
        <p>
          Avoid uploading sensitive images. Use test images when possible.
        </p>
      </div>
    </section>
  )
}

function ContactPage() {
  return (
    <section className="grid2">
      <div className="card card--pad">
        <div className="sectionHead">
          <div>
            <div className="sectionHead__title">Contact</div>
            <div className="sectionHead__sub">Questions, bugs, or feedback.</div>
          </div>
        </div>

        <div className="prose">
          <p>
            Add your preferred contact method here (email, form endpoint, or issue tracker).
          </p>
          <p>
            Quick option:{" "}
            <a className="link" href="mailto:team@example.com?subject=TRUST-AGENT%20Feedback">
              team@example.com
            </a>
          </p>
        </div>
      </div>

      <div className="card card--pad">
        <div className="card__title">What to include</div>
        <ul className="list">
          <li>Claim text you used</li>
          <li>Screenshot of results</li>
          <li>Expected vs actual behavior</li>
          <li>Console/server logs (if any)</li>
        </ul>
      </div>
    </section>
  )
}

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/contact" element={<ContactPage />} />
      </Routes>
    </Shell>
  )
}
