# title: Diet
# about: Minimize cost while hitting calorie and protein targets.
Minimize
  2 bread + 3 milk + 5 eggs
Subject To
  calories:  80 bread + 120 milk + 70 eggs >= 2000
  protein:   4 bread + 8 milk + 6 eggs >= 50
Bounds
  bread >= 0
  milk >= 0
  eggs >= 0
End
