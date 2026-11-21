// Light / Dark / Auto switch shared by the scanner (index.html) and the help page (help.html).
// The saved choice is also applied by a small inline script in each page's <head>, before first
// paint, so the page never flashes the wrong colours.

(function () {
  function apply(choice) {
    if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
    else delete document.documentElement.dataset.theme;
    document.querySelectorAll("[data-theme-choice]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themeChoice === choice)));
    try { localStorage.setItem("theme", choice); } catch (e) { /* private mode: the choice lasts this visit only */ }
  }

  function saved() {
    try { return localStorage.getItem("theme") || "auto"; } catch (e) { return "auto"; }
  }

  function init() {
    document.querySelectorAll("[data-theme-choice]").forEach((b) => b.addEventListener("click", () => apply(b.dataset.themeChoice)));
    apply(saved());
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
