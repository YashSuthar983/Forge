# title: Mini crude blend
# about: Mix two crudes into one product under a sulfur cap and crude capacity.
Maximize
  90 product - 40 light - 55 heavy
Subject To
  mass:     light + heavy - product = 0
  sulfur:   0.5 light + 2.5 heavy - 1.0 product <= 0
  demand:   product >= 100
  crude:    light + heavy <= 500
Bounds
  light >= 0
  heavy >= 0
  product >= 0
End
