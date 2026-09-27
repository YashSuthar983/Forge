# title: Knapsack
# about: Pick whole items that fit a weight limit - introduces Binary / General.
Maximize
  10 laptop + 6 camera + 4 book
Subject To
  weight: 5 laptop + 3 camera + 1 book <= 8
Binary
  laptop
  camera
  book
End
