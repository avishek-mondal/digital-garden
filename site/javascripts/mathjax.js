window.MathJax = {
  loader: {load: ['[tex]/ams', '[tex]/newcommand', '[tex]/extpfeil']},
  tex: {
    packages: {'[+]': ['ams', 'newcommand', 'extpfeil']},
    inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      macros: {
        // smash: ["\\mathllap{#1}", 1]
        mathrlap: ["\\mathchoice{\\rlap{\\displaystyle{#1}}}{\\rlap{\\textstyle{#1}}}{\\rlap{\\scriptstyle{#1}}}{\\rlap{\\scriptscriptstyle{#1}}}", 1],
        mathllap: ["\\mathchoice{\\llap{\\displaystyle{#1}}}{\\llap{\\textstyle{#1}}}{\\llap{\\scriptstyle{#1}}}{\\llap{\\scriptscriptstyle{#1}}}", 1],
        mathclap: ["\\mathchoice{\\clap{\\displaystyle{#1}}}{\\clap{\\textstyle{#1}}}{\\clap{\\scriptstyle{#1}}}{\\clap{\\scriptscriptstyle{#1}}}", 1],
        mysmash: ["\\mathrlap{\\phantom{#1}}#1", 1]
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
  