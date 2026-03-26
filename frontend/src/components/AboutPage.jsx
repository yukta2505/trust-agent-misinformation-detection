export default function AboutPage() {
  return (
    <main className="max-w-5xl mx-auto px-4 md:px-8 py-12 transition-colors duration-300">
      <div className="bg-white rounded-3xl shadow-sm border border-gray-200 p-8 md:p-12 dark:bg-[#111827] dark:border-slate-700">
        <h2 className="text-4xl font-semibold mb-6">About TRUST-AGENT</h2>

        <section className="mb-8">
          <h3 className="text-2xl font-semibold mb-2">Problem</h3>
          <p className="text-gray-600 leading-7 dark:text-gray-400">
            The rapid growth of social media has led to a significant rise in misinformation,
            especially through the misuse of authentic images presented with misleading claims
            about different times, locations, or events. These out-of-context images are often
            not detected by existing systems, which primarily focus on manipulated or fake visuals.
          </p>
        </section>

        <section className="mb-8">
          <h3 className="text-2xl font-semibold mb-2">Objectives</h3>
          <ul className="list-disc pl-6 text-gray-600 space-y-2 dark:text-gray-400">
            <li>Detect out-of-context misinformation using image–text consistency.</li>
            <li>Verify temporal and contextual alignment between image and claim.</li>
            <li>Assess credibility of information sources.</li>
            <li>Develop a multi-agent AI system mimicking human fact-checking.</li>
            <li>Provide human-readable explanations with final verdict.</li>
          </ul>
        </section>

        <section className="mb-8">
          <h3 className="text-2xl font-semibold mb-2">Methodology</h3>
          <p className="text-gray-600 leading-7 dark:text-gray-400">
            The system accepts an image and its associated claim as input. It retrieves
            historical evidence using reverse image search and web sources. This evidence
            is processed using vision-language models to understand image context.
            Multiple AI agents independently evaluate entities, temporal consistency,
            and source credibility.
          </p>
        </section>
      </div>
    </main>
  );
}