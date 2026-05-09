import { useState, useRef } from "react"

export default function UploadForm({ onSubmit, loading }) {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [claim, setClaim] = useState("")
  const fileRef = useRef()

  function handleFile(e) {
    const file = e.target.files[0]
    if (!file) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (!image || !claim.trim()) return
    onSubmit(image, claim)
  }

  const inputStyle = {
    background: "var(--surface)", border: "1px solid var(--border)",
    color: "var(--text)", borderRadius: 8, padding: "0.75rem 1rem",
    fontSize: "0.95rem", width: "100%", fontFamily: "inherit"
  }
  const btnStyle = {
    background: loading ? "var(--border)" : "var(--accent)",
    color: "#fff", border: "none", borderRadius: 8, padding: "0.75rem 2rem",
    fontSize: "1rem", fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
    marginTop: "1rem", fontFamily: "inherit"
  }

  return (
    <form onSubmit={handleSubmit}
          style={{ background: "var(--surface)", borderRadius: 12,
                   padding: "1.5rem", border: "1px solid var(--border)" }}>
      <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap" }}>
        {/* Image upload */}
        <div style={{ flex: 1, minWidth: 200 }}>
          <label style={{ display: "block", marginBottom: "0.5rem",
                          color: "var(--muted)", fontSize: "0.85rem" }}>
            IMAGE
          </label>
          <div onClick={() => fileRef.current.click()}
               style={{ ...inputStyle, cursor: "pointer", textAlign: "center",
                         padding: "1.5rem", border: "2px dashed var(--border)" }}>
            {preview
              ? <img src={preview} alt="preview"
                     style={{ maxHeight: 120, borderRadius: 6 }} />
              : <span style={{ color: "var(--muted)" }}>Click to upload image</span>
            }
          </div>
          <input ref={fileRef} type="file" accept="image/*"
                 onChange={handleFile} style={{ display: "none" }} />
        </div>

        {/* Claim text */}
        <div style={{ flex: 2, minWidth: 280 }}>
          <label style={{ display: "block", marginBottom: "0.5rem",
                          color: "var(--muted)", fontSize: "0.85rem" }}>
            CLAIM / CAPTION BEING FACT-CHECKED
          </label>
          <textarea
            value={claim} onChange={e => setClaim(e.target.value)}
            placeholder="e.g. This photo shows protesters in New Delhi, 2024..."
            rows={5} style={{ ...inputStyle, resize: "vertical" }}
          />
        </div>
      </div>

      <button type="submit" disabled={loading || !image || !claim.trim()}
              style={btnStyle}>
        {loading ? "Analysing..." : "Analyse"}
      </button>
    </form>
  )
}