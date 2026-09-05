*SENSE:Maximize
NAME          Refinery_Optimization
ROWS
 N  OBJ
 L  crude_avail_AL
 L  crude_avail_WTI
 L  crude_avail_Maya
 L  crude_avail_Brent
 L  cdu_capacity
 L  ln_balance_AL
 L  hn_balance_AL
 L  kero_balance_AL
 L  diesel_balance_AL
 L  ln_balance_WTI
 L  hn_balance_WTI
 L  kero_balance_WTI
 L  diesel_balance_WTI
 L  ln_balance_Maya
 L  hn_balance_Maya
 L  kero_balance_Maya
 L  diesel_balance_Maya
 L  ln_balance_Brent
 L  hn_balance_Brent
 L  kero_balance_Brent
 L  diesel_balance_Brent
 L  reformer_capacity
 E  reformate_yield
 L  htu_capacity
 E  htu_output_balance
 E  gasoline_blend_balance
 E  diesel_blend_balance
 E  jet_blend_balance
 L  lpg_balance
 L  residue_balance
 L  diesel_sulfur_spec
COLUMNS
    crude_buy_AL  crude_avail_AL   1.000000000000e+00
    crude_buy_AL  cdu_capacity   1.000000000000e+00
    crude_buy_AL  ln_balance_AL  -6.000000000000e-02
    crude_buy_AL  hn_balance_AL  -1.300000000000e-01
    crude_buy_AL  kero_balance_AL  -1.450000000000e-01
    crude_buy_AL  diesel_balance_AL  -1.950000000000e-01
    crude_buy_AL  lpg_balance  -2.000000000000e-02
    crude_buy_AL  residue_balance  -4.500000000000e-01
    crude_buy_AL  OBJ       -6.750000000000e+01
    crude_buy_Brent  crude_avail_Brent   1.000000000000e+00
    crude_buy_Brent  cdu_capacity   1.000000000000e+00
    crude_buy_Brent  ln_balance_Brent  -8.000000000000e-02
    crude_buy_Brent  hn_balance_Brent  -1.550000000000e-01
    crude_buy_Brent  kero_balance_Brent  -1.600000000000e-01
    crude_buy_Brent  diesel_balance_Brent  -2.250000000000e-01
    crude_buy_Brent  lpg_balance  -2.500000000000e-02
    crude_buy_Brent  residue_balance  -3.550000000000e-01
    crude_buy_Brent  OBJ       -7.400000000000e+01
    crude_buy_Maya  crude_avail_Maya   1.000000000000e+00
    crude_buy_Maya  cdu_capacity   1.000000000000e+00
    crude_buy_Maya  ln_balance_Maya  -3.000000000000e-02
    crude_buy_Maya  hn_balance_Maya  -8.500000000000e-02
    crude_buy_Maya  kero_balance_Maya  -1.150000000000e-01
    crude_buy_Maya  diesel_balance_Maya  -1.650000000000e-01
    crude_buy_Maya  lpg_balance  -1.000000000000e-02
    crude_buy_Maya  residue_balance  -5.950000000000e-01
    crude_buy_Maya  OBJ       -5.950000000000e+01
    crude_buy_WTI  crude_avail_WTI   1.000000000000e+00
    crude_buy_WTI  cdu_capacity   1.000000000000e+00
    crude_buy_WTI  ln_balance_WTI  -8.500000000000e-02
    crude_buy_WTI  hn_balance_WTI  -1.650000000000e-01
    crude_buy_WTI  kero_balance_WTI  -1.650000000000e-01
    crude_buy_WTI  diesel_balance_WTI  -2.350000000000e-01
    crude_buy_WTI  lpg_balance  -3.000000000000e-02
    crude_buy_WTI  residue_balance  -3.200000000000e-01
    crude_buy_WTI  OBJ       -7.150000000000e+01
    hn_to_reformer_AL  hn_balance_AL   1.000000000000e+00
    hn_to_reformer_AL  reformer_capacity   1.000000000000e+00
    hn_to_reformer_AL  reformate_yield  -8.500000000000e-01
    hn_to_reformer_AL  lpg_balance  -5.000000000000e-02
    hn_to_reformer_AL  OBJ       -2.500000000000e+00
    hn_to_reformer_Brent  hn_balance_Brent   1.000000000000e+00
    hn_to_reformer_Brent  reformer_capacity   1.000000000000e+00
    hn_to_reformer_Brent  reformate_yield  -8.500000000000e-01
    hn_to_reformer_Brent  lpg_balance  -5.000000000000e-02
    hn_to_reformer_Brent  OBJ       -2.500000000000e+00
    hn_to_reformer_Maya  hn_balance_Maya   1.000000000000e+00
    hn_to_reformer_Maya  reformer_capacity   1.000000000000e+00
    hn_to_reformer_Maya  reformate_yield  -8.500000000000e-01
    hn_to_reformer_Maya  lpg_balance  -5.000000000000e-02
    hn_to_reformer_Maya  OBJ       -2.500000000000e+00
    hn_to_reformer_WTI  hn_balance_WTI   1.000000000000e+00
    hn_to_reformer_WTI  reformer_capacity   1.000000000000e+00
    hn_to_reformer_WTI  reformate_yield  -8.500000000000e-01
    hn_to_reformer_WTI  lpg_balance  -5.000000000000e-02
    hn_to_reformer_WTI  OBJ       -2.500000000000e+00
    product_sale_diesel  diesel_blend_balance   1.000000000000e+00
    product_sale_diesel  diesel_sulfur_spec  -1.500000000000e+01
    product_sale_diesel  OBJ        1.000000000000e+02
    product_sale_gasoline  gasoline_blend_balance   1.000000000000e+00
    product_sale_gasoline  OBJ        9.500000000000e+01
    product_sale_jet  jet_blend_balance   1.000000000000e+00
    product_sale_jet  OBJ        9.200000000000e+01
    product_sale_lpg  lpg_balance   1.000000000000e+00
    product_sale_lpg  OBJ        5.000000000000e+01
    product_sale_residue_fuel  residue_balance   1.000000000000e+00
    product_sale_residue_fuel  OBJ        6.000000000000e+01
    to_diesel_diesel_AL  diesel_balance_AL   1.000000000000e+00
    to_diesel_diesel_AL  diesel_blend_balance  -1.000000000000e+00
    to_diesel_diesel_AL  diesel_sulfur_spec   1.100000000000e+04
    to_diesel_diesel_Brent  diesel_balance_Brent   1.000000000000e+00
    to_diesel_diesel_Brent  diesel_blend_balance  -1.000000000000e+00
    to_diesel_diesel_Brent  diesel_sulfur_spec   2.200000000000e+03
    to_diesel_diesel_Maya  diesel_balance_Maya   1.000000000000e+00
    to_diesel_diesel_Maya  diesel_blend_balance  -1.000000000000e+00
    to_diesel_diesel_Maya  diesel_sulfur_spec   2.200000000000e+04
    to_diesel_diesel_WTI  diesel_balance_WTI   1.000000000000e+00
    to_diesel_diesel_WTI  diesel_blend_balance  -1.000000000000e+00
    to_diesel_diesel_WTI  diesel_sulfur_spec   1.500000000000e+03
    to_diesel_htu_diesel  htu_output_balance   1.000000000000e+00
    to_diesel_htu_diesel  diesel_blend_balance  -1.000000000000e+00
    to_diesel_htu_diesel  diesel_sulfur_spec   1.000000000000e+01
    to_gasoline_ln_AL  ln_balance_AL   1.000000000000e+00
    to_gasoline_ln_AL  gasoline_blend_balance  -1.000000000000e+00
    to_gasoline_ln_Brent  ln_balance_Brent   1.000000000000e+00
    to_gasoline_ln_Brent  gasoline_blend_balance  -1.000000000000e+00
    to_gasoline_ln_Maya  ln_balance_Maya   1.000000000000e+00
    to_gasoline_ln_Maya  gasoline_blend_balance  -1.000000000000e+00
    to_gasoline_ln_WTI  ln_balance_WTI   1.000000000000e+00
    to_gasoline_ln_WTI  gasoline_blend_balance  -1.000000000000e+00
    to_gasoline_reformate  reformate_yield   1.000000000000e+00
    to_gasoline_reformate  gasoline_blend_balance  -1.000000000000e+00
    to_htu_AL  diesel_balance_AL   1.000000000000e+00
    to_htu_AL  htu_capacity   1.000000000000e+00
    to_htu_AL  htu_output_balance  -1.000000000000e+00
    to_htu_AL  OBJ       -1.200000000000e+00
    to_htu_Brent  diesel_balance_Brent   1.000000000000e+00
    to_htu_Brent  htu_capacity   1.000000000000e+00
    to_htu_Brent  htu_output_balance  -1.000000000000e+00
    to_htu_Brent  OBJ       -1.200000000000e+00
    to_htu_Maya  diesel_balance_Maya   1.000000000000e+00
    to_htu_Maya  htu_capacity   1.000000000000e+00
    to_htu_Maya  htu_output_balance  -1.000000000000e+00
    to_htu_Maya  OBJ       -1.200000000000e+00
    to_htu_WTI  diesel_balance_WTI   1.000000000000e+00
    to_htu_WTI  htu_capacity   1.000000000000e+00
    to_htu_WTI  htu_output_balance  -1.000000000000e+00
    to_htu_WTI  OBJ       -1.200000000000e+00
    to_jet_kero_AL  kero_balance_AL   1.000000000000e+00
    to_jet_kero_AL  jet_blend_balance  -1.000000000000e+00
    to_jet_kero_Brent  kero_balance_Brent   1.000000000000e+00
    to_jet_kero_Brent  jet_blend_balance  -1.000000000000e+00
    to_jet_kero_Maya  kero_balance_Maya   1.000000000000e+00
    to_jet_kero_Maya  jet_blend_balance  -1.000000000000e+00
    to_jet_kero_WTI  kero_balance_WTI   1.000000000000e+00
    to_jet_kero_WTI  jet_blend_balance  -1.000000000000e+00
RHS
    RHS       crude_avail_AL   8.000000000000e+04
    RHS       crude_avail_WTI   6.000000000000e+04
    RHS       crude_avail_Maya   7.000000000000e+04
    RHS       crude_avail_Brent   5.000000000000e+04
    RHS       cdu_capacity   1.000000000000e+05
    RHS       ln_balance_AL   0.000000000000e+00
    RHS       hn_balance_AL   0.000000000000e+00
    RHS       kero_balance_AL   0.000000000000e+00
    RHS       diesel_balance_AL   0.000000000000e+00
    RHS       ln_balance_WTI   0.000000000000e+00
    RHS       hn_balance_WTI   0.000000000000e+00
    RHS       kero_balance_WTI   0.000000000000e+00
    RHS       diesel_balance_WTI   0.000000000000e+00
    RHS       ln_balance_Maya   0.000000000000e+00
    RHS       hn_balance_Maya   0.000000000000e+00
    RHS       kero_balance_Maya   0.000000000000e+00
    RHS       diesel_balance_Maya   0.000000000000e+00
    RHS       ln_balance_Brent   0.000000000000e+00
    RHS       hn_balance_Brent   0.000000000000e+00
    RHS       kero_balance_Brent   0.000000000000e+00
    RHS       diesel_balance_Brent   0.000000000000e+00
    RHS       reformer_capacity   2.000000000000e+04
    RHS       reformate_yield   0.000000000000e+00
    RHS       htu_capacity   3.000000000000e+04
    RHS       htu_output_balance   0.000000000000e+00
    RHS       gasoline_blend_balance   0.000000000000e+00
    RHS       diesel_blend_balance   0.000000000000e+00
    RHS       jet_blend_balance   0.000000000000e+00
    RHS       lpg_balance   0.000000000000e+00
    RHS       residue_balance   0.000000000000e+00
    RHS       diesel_sulfur_spec   0.000000000000e+00
BOUNDS
ENDATA
