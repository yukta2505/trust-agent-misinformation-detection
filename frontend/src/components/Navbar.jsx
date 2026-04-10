import { Activity, Moon, Sun } from "lucide-react";

export default function Navbar({
  activeTab,
  setActiveTab,
  theme,
  setTheme,
}) {
  const navBtn = (id, label) => {
    const active = activeTab === id;

    return (
      <button
        onClick={() => setActiveTab(id)}
        className={`px-6 py-2 rounded-2xl text-sm md:text-base font-medium transition-all duration-200 ${
          active
            ? "bg-blue-900 text-white shadow-md dark:bg-white dark:text-[#0B1120]"
            : "text-gray-600 hover:bg-white hover:shadow-sm dark:text-gray-300 dark:hover:bg-[#1E293B]"
        }`}
      >
        {label}
      </button>
    );
  };

  return (
    <header className="sticky top-0 z-50 bg-white/90 backdrop-blur border-b border-gray-200 dark:bg-[#0F172A]/90 dark:border-slate-700 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 md:px-8 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold tracking-wide">
            TRUST-AGENT
          </h1>
          <p className="text-gray-500 tracking-[0.25em] text-xs md:text-sm mt-1 dark:text-gray-400">
            Misinformation Detection
          </p>
        </div>

        <div className="hidden md:flex items-center gap-3">
          {navBtn("analyze", "Analyze")}
          {navBtn("how", "How It Works")}
          {navBtn("about", "About")}
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            className="p-2 rounded-xl border border-gray-200 bg-white hover:bg-gray-100 dark:bg-[#1E293B] dark:border-slate-600 dark:hover:bg-[#334155] transition"
          >
            {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
          </button>

          <div className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <Activity size={18} />
            <span className="hidden sm:inline">System Online</span>
          </div>
        </div>
      </div>

      <div className="md:hidden px-4 pb-4 flex gap-2 justify-center">
        {navBtn("analyze", "Analyze")}
        {navBtn("how", "How It Works")}
        {navBtn("about", "About")}
      </div>
    </header>
  );
}