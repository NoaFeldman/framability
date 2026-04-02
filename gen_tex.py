import sys, io

sys.stdout = io.StringIO()
from Dicke import Lindbladian_superop_symbolic
import sympy as sp

L_sym, labels = Lindbladian_superop_symbolic(3)
sys.stdout = sys.__stdout__

tex = sp.latex(L_sym)

with open('Lindbladian_N3.tex', 'w', encoding='utf-8') as f:
    f.write('\\documentclass{article}\n')
    f.write('\\usepackage{amsmath}\n')
    f.write('\\usepackage[landscape,margin=0.5cm]{geometry}\n')
    f.write('\\begin{document}\n')
    f.write('\\tiny\n')
    f.write('$$\n')
    f.write(tex + '\n')
    f.write('$$\n')
    f.write('\\end{document}\n')

print('Saved to Lindbladian_N3.tex')
