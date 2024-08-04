window.MathJax = {
  loader: {load: ['[tex]/ams', '[tex]/newcommand', '[tex]/extpfeil', '[tex]/color', '[tex]/physics']},
  tex: {
    packages: {'[+]': ['ams', 'newcommand', 'extpfeil', 'color', 'physics']},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      whitetextemdash: ["\\textcolor{white}{\\raise0.5ex{\\rule{1cm}{0.4pt}}}", 0]

    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
})