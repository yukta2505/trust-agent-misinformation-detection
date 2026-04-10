export default function HowItWorksPage() {
  const items = [
    {
      title: "Upload & Input",
      text: "Submit an image along with the associated text claim for verification.",
    },
    {
      title: "Context & Entity Agent",
      text: "Extracts entities, generates captions, and performs reverse image search to identify geographic and semantic mismatches.",
    },
    {
      title: "Temporal Consistency Agent",
      text: "Analyzes EXIF metadata, publication dates, and historical context to verify temporal accuracy.",
    },
    {
      title: "Source Credibility Agent",
      text: "Evaluates source reliability through cross-referencing with credible databases and fact-check archives.",
    },
    {
      title: "Final Verdict",
      text: "Integrates all agent outputs using weighted decision logic to produce an explainable, transparent verdict.",
    },
  ];

  return (
    <main className="max-w-7xl mx-auto px-4 md:px-8 py-10">
      <section className="text-center mb-12">
        <h2 className="text-4xl font-semibold mb-3">How It Works</h2>
        <p className="text-gray-500 text-xl">
          A multi-agent pipeline that mirrors professional fact-checking methodology.
        </p>
      </section>

      <div className="space-y-10">
        {items.map((item, idx) => (
          <div key={idx} className="border-l-4 border-gray-200 pl-6">
            <h3 className="text-3xl font-semibold mb-2">{item.title}</h3>
            <p className="text-gray-500 text-xl leading-relaxed">{item.text}</p>
          </div>
        ))}
      </div>
    </main>
  );
}