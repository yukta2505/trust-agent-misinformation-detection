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

export default function AnalyzePage() {
  const [file, setFile] = useState(null);
  const [claim, setClaim] = useState("");
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const inputRef = useRef(null);

  const analysisSteps = [
    { title: "Context Analysis", icon: Eye },
    { title: "Temporal Check", icon: Search },
    { title: "Source Verification", icon: Shield },
  ];

  const mockResult = {
    verdict: "Pristine",
    confidence: 94,
    context: 95,
    temporal: 97,
    credibility: 92,
    explanations: [
      "Verified as showing Mumbai during July monsoon based on matching landmarks and weather conditions.",
      "Temporal evidence aligns with historical flood coverage and metadata consistency.",
      "Trusted news and archival sources reference similar visuals in the same context.",
    ],
    evidence: [
      {
        source: "Times of India",
        title: "Mumbai monsoon flooding coverage archive",
        snippet: "Archived reporting confirms recurring flood visuals from Mumbai monsoon periods.",
      },
      {
        source: "Hindustan Times",
        title: "Historical flood image reference",
        snippet: "Published article references the same location and weather conditions.",
      },
    ],
  };

  const handleBrowse = () => {
    inputRef.current?.click();
  };

  const handleSelectedFile = (selected) => {
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    handleSelectedFile(selected);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const selected = e.dataTransfer.files[0];
    handleSelectedFile(selected);
  };

  const handleAnalyze = () => {
    if (!file || !claim.trim()) return;
    setLoading(true);
    setShowResults(false);

    setTimeout(() => {
      setLoading(false);
      setShowResults(true);
    }, 2000);
  };

  const scoreCard = (title, value, subtitle, icon) => {
    const Icon = icon;
    return (
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
          <span className="px-4 py-1 rounded-xl text-sm font-medium bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300">
            MATCH
          </span>
        </div>
      </div>
    );
  };

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
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
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
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {file?.name}
              </p>
            </div>
          ) : (
            <>
              <Upload size={42} className="mx-auto text-gray-500 mb-4 dark:text-gray-400" />
              <p className="text-2xl md:text-3xl font-medium">
                Drop image here or browse
              </p>
              <p className="text-gray-500 mt-2 dark:text-gray-400">
                PNG, JPG, WEBP up to 10MB
              </p>
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
      </section>

      {/* Results */}
      {showResults && (
        <section className="mt-10 space-y-6">
          {/* Final Verdict */}
          <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:bg-[#111827] dark:border-slate-700">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div className="flex items-center gap-3">
                {mockResult.verdict === "Pristine" ? (
                  <CheckCircle2 className="text-emerald-600" size={30} />
                ) : (
                  <AlertTriangle className="text-red-500" size={30} />
                )}
                <div className="flex items-center gap-3 flex-wrap">
                  <h3 className="text-3xl font-semibold">Final Verdict</h3>
                  <span className={`px-4 py-1 rounded-xl text-lg font-medium ${
                    mockResult.verdict === "Pristine"
                      ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300"
                      : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
                  }`}>
                    <BadgeCheck className="inline mr-2" size={18} />
                    {mockResult.verdict}
                  </span>
                </div>
              </div>

              <p className="text-2xl font-semibold">
                Confidence: {mockResult.confidence}%
              </p>
            </div>

            <div className="mt-6 h-4 w-full rounded-full bg-gray-200 dark:bg-slate-700 overflow-hidden">
              <div
                className="h-full bg-emerald-500 rounded-full"
                style={{ width: `${mockResult.confidence}%` }}
              />
            </div>
          </div>

          {/* Score Cards */}
          <div className="grid md:grid-cols-3 gap-6">
            {scoreCard("Context Score", mockResult.context, "Matches location and scene context", Eye)}
            {scoreCard("Temporal Score", mockResult.temporal, "Consistent with historical timing", Search)}
            {scoreCard("Credibility Score", mockResult.credibility, "Supported by reliable sources", Shield)}
          </div>

          {/* Explanations */}
          <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:bg-[#111827] dark:border-slate-700">
            <h3 className="text-2xl font-semibold mb-6">Explanations</h3>
            <div className="space-y-4">
              {mockResult.explanations.map((item, idx) => (
                <div key={idx} className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-emerald-500 text-white flex items-center justify-center font-semibold shrink-0">
                    {idx + 1}
                  </div>
                  <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed">
                    {item}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Evidence */}
          <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm dark:bg-[#111827] dark:border-slate-700">
            <h3 className="text-2xl font-semibold mb-6">Evidence</h3>
            <div className="grid md:grid-cols-2 gap-4">
              {mockResult.evidence.map((item, idx) => (
                <div
                  key={idx}
                  className="rounded-2xl border border-gray-200 p-5 bg-gray-50 dark:bg-[#0F172A] dark:border-slate-700"
                >
                  <p className="text-sm font-medium text-blue-900 dark:text-blue-300 mb-2">
                    {item.source}
                  </p>
                  <h4 className="text-xl font-semibold mb-2">{item.title}</h4>
                  <p className="text-gray-600 dark:text-gray-400">{item.snippet}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </main>
  );
}