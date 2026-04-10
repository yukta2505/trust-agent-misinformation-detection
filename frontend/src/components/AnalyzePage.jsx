import { useRef, useState } from "react";
import {
  Eye,
  Search,
  Shield,
  Upload,
  Image as ImageIcon,
  FileText,
  CheckCircle2,
  AlertTriangle,
  BadgeCheck,
  FileSearch,
  Loader2,
  ChevronRight,
} from "lucide-react";

const API = "http://localhost:8000";

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [claim, setClaim] = useState("");
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const inputRef = useRef(null);

  const analysisSteps = [
    { title: "Context Analysis", icon: Eye },
    { title: "Temporal Check", icon: Search },
    { title: "Source Verification", icon: Shield },
  ];

  const handleBrowse = () => inputRef.current?.click();

  const handleSelectedFile = (selected) => {
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleFileChange = (e) => handleSelectedFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleSelectedFile(e.dataTransfer.files[0]);
  };

  const handleAnalyze = async () => {
    if (!file || !claim.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const form = new FormData();
    form.append("image", file);
    form.append("claim", claim);

    try {
      const res = await fetch(`${API}/analyse`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to the analysis server.");
    } finally {
      setLoading(false);
    }
  };

  // Derive display values from real API response
  // API returns: verdict, confidence_percent, entity_score, temporal_score,
  //              credibility_score, final_score, explanation, caption,
  //              flags, key_evidence_for_verdict, evidence[]
  const isOOC = result?.verdict === "OUT-OF-CONTEXT";
  const verdictLabel = result
    ? isOOC
      ? "Out-of-Context"
      : "Pristine"
    : null;

  const scoreCard = (title, value, subtitle, Icon) => (
    <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:bg-[#111827] dark:border-slate-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Icon className="text-blue-900 dark:text-blue-300" size={24} />
          <h3 className="text-xl font-semibold">{title}</h3>
        </div>
        <ChevronRight className="text-gray-400" size={20} />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-5xl font-bold">{value}%</p>
          <p className="text-gray-500 mt-3 dark:text-gray-400">{subtitle}</p>
        </div>
        <span className={`px-4 py-1 rounded-xl text-sm font-medium ${
          value >= 70
            ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
            : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
        }`}>
          {value >= 70 ? "MATCH" : "MISMATCH"}
        </span>
      </div>
    </div>
  );

  return (
    <main className="max-w-7xl mx-auto px-4 md:px-8 py-10 transition-colors duration-300">
      {/* Hero */}
      <section className="text-center mb-10">
        <p className="inline-flex items-center gap-2 text-sm font-medium text-gray-700 mb-4 dark:text-gray-300">
          ⚡ Multi-Agent AI Verification
        </p>
        <h2 className="text-3xl md:text-5xl font-semibold leading-tight">
          Detect Out-of-Context
          <br />
          Misinformation
        </h2>
        <p className="mt-5 text-gray-500 text-lg max-w-3xl mx-auto dark:text-gray-400">
          Upload an image and its associated claim. Our AI agents will verify
          context, temporal consistency, and source credibility.
        </p>
      </section>

      {/* Analysis Cards */}
      <section className="grid grid-cols-1 gap-0 border border-gray-200 rounded-3xl overflow-hidden bg-white shadow-sm dark:bg-[#111827] dark:border-slate-700">
        {analysisSteps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div
              key={idx}
              className={`flex flex-col items-center justify-center py-12 ${
                idx !== analysisSteps.length - 1
                  ? "border-b border-gray-200 dark:border-slate-700"
                  : ""
              }`}
            >
              <Icon size={34} className="text-gray-700 mb-4 dark:text-gray-300" />
              <h3 className="text-2xl font-medium">{step.title}</h3>
            </div>
          );
        })}
      </section>

      {/* Upload Section */}
      <section className="mt-8 border border-gray-200 rounded-3xl bg-white shadow-sm p-6 md:p-8 dark:bg-[#111827] dark:border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <ImageIcon size={20} />
          <h3 className="text-2xl font-semibold">Upload Image</h3>
        </div>

        <div
          onClick={handleBrowse}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-2xl p-6 md:p-10 text-center cursor-pointer transition ${
            dragActive
              ? "border-blue-600 bg-blue-50 dark:bg-blue-900/10"
              : "border-gray-300 hover:bg-gray-50 dark:border-slate-600 dark:hover:bg-[#1E293B]"
          }`}
        >
          {preview ? (
            <div className="space-y-4">
              <img
                src={preview}
                alt="Preview"
                className="mx-auto rounded-2xl max-h-[400px] object-cover shadow"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400">{file?.name}</p>
            </div>
          ) : (
            <>
              <Upload size={42} className="mx-auto text-gray-500 mb-4 dark:text-gray-400" />
              <p className="text-2xl md:text-3xl font-medium">Drop image here or browse</p>
              <p className="text-gray-500 mt-2 dark:text-gray-400">PNG, JPG, WEBP up to 10MB</p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.webp"
            className="hidden"
            onChange={handleFileChange}
          />
        </div>

        <div className="mt-8">
          <div className="flex items-center gap-2 mb-3">
            <FileText size={20} />
            <h3 className="text-2xl font-semibold">Text Claim</h3>
          </div>
          <textarea
            value={claim}
            onChange={(e) => setClaim(e.target.value)}
            maxLength={500}
            rows={4}
            placeholder='e.g. "This photo shows flooding in Mumbai during the 2024 monsoon season"'
            className="w-full border border-gray-200 rounded-2xl p-5 text-lg resize-none bg-gray-50 focus:bg-white focus:border-blue-900 transition dark:bg-[#0F172A] dark:border-slate-700 dark:focus:border-blue-400 dark:text-white"
          />
          <div className="flex justify-end mt-2 text-sm text-gray-500 dark:text-gray-400">
            {claim.length}/500
          </div>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={!file || !claim.trim() || loading}
          className="mt-6 w-full rounded-2xl bg-blue-900 text-white py-4 text-xl font-medium hover:opacity-95 transition disabled:opacity-50 dark:bg-white dark:text-[#0B1120] flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" size={22} />
              Analyzing...
            </>
          ) : (
            <>
              <FileSearch size={22} />
              Analyze for Misinformation
            </>
          )}
        </button>

        {/* Loading sub-label */}
        {loading && (
          <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-3">
            Entity · Temporal · Credibility agents running in parallel…
          </p>
        )}
      </section>

      {/* Error */}
      {error && (
        <div className="mt-6 rounded-2xl border border-red-300 bg-red-50 dark:bg-red-900/20 dark:border-red-700 p-5 text-red-700 dark:text-red-300">
          <AlertTriangle className="inline mr-2" size={18} />
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <section className="mt-10 space-y-6">

          {/* Final Verdict */}
          <div className={`rounded-3xl border-2 bg-white p-6 shadow-sm dark:bg-[#111827] ${
            isOOC ? "border-red-400 dark:border-red-600" : "border-emerald-400 dark:border-emerald-600"
          }`}>
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center gap-3 flex-wrap">
                {isOOC
                  ? <AlertTriangle className="text-red-500" size={30} />
                  : <CheckCircle2 className="text-emerald-600" size={30} />
                }
                <h3 className="text-3xl font-semibold">Final Verdict</h3>
                <span className={`px-4 py-1 rounded-xl text-lg font-medium ${
                  isOOC
                    ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                    : "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                }`}>
                  <BadgeCheck className="inline mr-2" size={18} />
                  {verdictLabel}
                </span>
              </div>
              <p className="text-2xl font-semibold">
                Confidence: {result.confidence_percent}%
              </p>
            </div>

            <div className="mt-6 h-4 w-full rounded-full bg-gray-200 dark:bg-slate-700 overflow-hidden">
              <div
                className={`h-full rounded-full ${isOOC ? "bg-red-500" : "bg-emerald-500"}`}
                style={{ width: `${result.confidence_percent}%` }}
              />
            </div>

            {/* Explanation */}
            {result.explanation && (
              <p className="mt-5 text-lg text-gray-700 dark:text-gray-300 leading-relaxed">
                {result.explanation}
              </p>
            )}

            {/* Caption */}
            {result.caption && (
              <div className="mt-4 rounded-xl bg-gray-100 dark:bg-[#0F172A] px-4 py-3 text-sm font-mono text-gray-600 dark:text-gray-400">
                Caption: {result.caption}
              </div>
            )}

            {/* Processing time */}
            {result.processing_time_sec !== undefined && (
              <p className="mt-3 text-sm text-gray-400 dark:text-gray-500">
                Processed in {result.processing_time_sec}s
              </p>
            )}
          </div>

          {/* Score Cards */}
          <div className="grid md:grid-cols-3 gap-6">
            {scoreCard(
              "Context Score",
              Math.round((result.entity_score ?? 0) * 100),
              "Matches location and scene context",
              Eye
            )}
            {scoreCard(
              "Temporal Score",
              Math.round((result.temporal_score ?? 0) * 100),
              "Consistent with historical timing",
              Search
            )}
            {scoreCard(
              "Credibility Score",
              Math.round((result.credibility_score ?? 0) * 100),
              "Supported by reliable sources",
              Shield
            )}
          </div>

          {/* Red Flags */}
          {result.flags?.length > 0 && (
            <div className="rounded-3xl border border-red-200 bg-red-50 dark:bg-red-900/10 dark:border-red-800 p-6 shadow-sm">
              <h3 className="text-2xl font-semibold mb-4 text-red-700 dark:text-red-300">
                Red Flags
              </h3>
              <div className="space-y-2">
                {result.flags.map((flag, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <AlertTriangle className="text-red-500 shrink-0 mt-1" size={18} />
                    <p className="text-red-700 dark:text-red-300">{flag}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Key Evidence for Verdict */}
          {result.key_evidence_for_verdict?.length > 0 && (
            <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:bg-[#111827] dark:border-slate-700">
              <h3 className="text-2xl font-semibold mb-6">Key Evidence</h3>
              <div className="space-y-4">
                {result.key_evidence_for_verdict.map((item, idx) => (
                  <div key={idx} className="flex gap-4">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center font-semibold shrink-0 text-white ${
                      isOOC ? "bg-red-500" : "bg-emerald-500"
                    }`}>
                      {idx + 1}
                    </div>
                    <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed">
                      {item}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Evidence List */}
          {result.evidence?.length > 0 && (
            <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:bg-[#111827] dark:border-slate-700">
              <h3 className="text-2xl font-semibold mb-2">
                Evidence
                <span className="ml-2 text-base font-normal text-gray-400">
                  ({result.evidence.length} sources)
                </span>
              </h3>
              <div className="grid md:grid-cols-2 gap-4 mt-6">
                {result.evidence.map((item, idx) => (
                  <div
                    key={idx}
                    className="rounded-2xl border border-gray-200 p-5 bg-gray-50 dark:bg-[#0F172A] dark:border-slate-700"
                  >
                    <p className="text-sm font-medium text-blue-900 dark:text-blue-300 mb-2">
                      {item.source}
                    </p>
                    <h4 className="text-lg font-semibold mb-2">
                      {item.title || "(no title)"}
                    </h4>
                    {item.snippet && (
                      <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed mb-3">
                        {item.snippet}
                      </p>
                    )}
                    <div className="flex items-center justify-between text-xs text-gray-400 dark:text-gray-500">
                      {item.score !== undefined && (
                        <span>Score: {item.score.toFixed(3)}</span>
                      )}
                      {item.url && (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-blue-600 dark:text-blue-400 hover:underline"
                        >
                          View source →
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
