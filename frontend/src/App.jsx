import { useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import AnalyzePage from "./components/AnalyzePage";
import HowItWorksPage from "./components/HowItWorksPage";
import AboutPage from "./components/AboutPage";

export default function App() {
  const [activeTab, setActiveTab] = useState("analyze");
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    const savedTheme = localStorage.getItem("trust-agent-theme");
    if (savedTheme) {
      setTheme(savedTheme);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("trust-agent-theme", theme);
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return (
    <div className="min-h-screen bg-slate-50 text-gray-900 dark:bg-[#0B1120] dark:text-white transition-colors duration-300">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
      />

      {activeTab === "analyze" && <AnalyzePage />}
      {activeTab === "how" && <HowItWorksPage />}
      {activeTab === "about" && <AboutPage />}
    </div>
  );
}