import React from "react"
import ReactDOM from "react-dom/client"
import { BrowserRouter } from "react-router-dom"
import App from "./App.jsx"
import "./index.css"

function initTheme() {
  const saved = localStorage.getItem("theme")
  const systemDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches
  const theme = saved || (systemDark ? "dark" : "light")
  document.documentElement.setAttribute("data-theme", theme)
}
initTheme()

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
