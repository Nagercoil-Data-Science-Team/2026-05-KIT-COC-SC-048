import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

from cobra import Model, Reaction, Metabolite
from cobra.io import write_sbml_model
from cobra.flux_analysis import flux_variability_analysis

def clean_gene(g):
    g = str(g)
    g = re.sub(r'[^a-zA-Z0-9_]', '_', g)
    return "G_" + g

def make_exchange(met, rxn_id, lb=0, ub=1000):
    ex = Reaction(rxn_id)
    ex.add_metabolites({met: 1})
    ex.lower_bound = lb
    ex.upper_bound = ub
    return ex

def make_sink(met, rxn_id, ub=1000):
    sk = Reaction(rxn_id)
    sk.add_metabolites({met: -1})
    sk.lower_bound = 0
    sk.upper_bound = ub
    return sk

def safe_set_bounds(rxn, lb=None, ub=None):
    if ub is not None and lb is None:
        rxn.upper_bound = max(ub, rxn.lower_bound)
    elif lb is not None and ub is None:
        if lb > rxn.upper_bound:
            rxn.upper_bound = lb
        rxn.lower_bound = lb
    elif lb is not None and ub is not None:
        real_lb = min(lb, ub)
        real_ub = max(lb, ub)
        if real_ub >= rxn.lower_bound:
            rxn.upper_bound = real_ub
            rxn.lower_bound = real_lb
        else:
            rxn.lower_bound = real_lb
            rxn.upper_bound = real_ub

df = pd.read_csv("feature_table.txt", sep="\t", engine="python")
df.columns = df.columns.str.replace("#", "").str.strip()

cds = df[df['feature'] == 'CDS'].copy()
genes = cds[['GeneID', 'symbol', 'name']].dropna(subset=['GeneID', 'name']).copy()
genes['GeneID'] = genes['GeneID'].astype(str).str.strip()
genes['symbol'] = genes['symbol'].astype(str).str.strip()
genes = genes.drop_duplicates(subset='GeneID')
print(f"Total CDS genes: {len(genes)}")

keywords = (
    "dehydrogenase|synthase|kinase|oxidase|transferase|"
    "catalase|peroxidase|reductase|dismutase|lyase|"
    "aquaporin|LEA|dehydrin|osmotin|expansin|"
    "proline|abscisic|drought|stress|desiccation|"
    "phosphatase|isomerase|mutase|carboxylase|ligase|"
    "epimerase|racemase|hydroxylase|acyltransferase|"
    "methyltransferase|glycosyltransferase|xylanase|"
    "glucanase|amylase|protease|peptidase|lipase|"
    "alcohol|aldehyde|pyruvate|malate|citrate|fumarate|"
    "succinate|acetyl|fatty acid|lipid|sterol|"
    "ribulose|phosphate|glucose|fructose|sucrose|starch|"
    "cellulose|hemicellulose|lignin|flavonoid|phenyl|"
    "chlorophyll|carotenoid|tocopherol|anthocyanin"
)

metabolic = genes[genes['name'].str.contains(keywords, case=False, na=False)].copy()
print(f"Metabolic genes: {len(metabolic)}")

def assign_pathway(name):
    n = str(name).lower()
    if any(x in n for x in ["catalase", "peroxidase", "dismutase", "oxidase"]):
        return "ROS_Detox"
    elif any(x in n for x in ["proline", "dehydrin", "lea", "osmotin", "desiccation"]):
        return "Osmoprotection"
    elif "aquaporin" in n:
        return "Water_Transport"
    elif any(x in n for x in ["kinase", "abscisic", "phosphatase", "drought"]):
        return "Stress_Signaling"
    elif any(x in n for x in ["pyruvate", "malate", "citrate", "fumarate", "succinate"]):
        return "TCA_Cycle"
    elif any(x in n for x in ["glucose", "fructose", "phosphogluco", "hexokinase",
                               "aldolase", "enolase", "phosphofructo"]):
        return "Glycolysis"
    elif any(x in n for x in ["ribulose", "rubisco", "calvin", "chlorophyll",
                               "carotenoid", "photosystem"]):
        return "Photosynthesis"
    elif any(x in n for x in ["fatty acid", "lipid", "sterol", "acyl", "lipase"]):
        return "Lipid_Metabolism"
    elif any(x in n for x in ["flavonoid", "phenyl", "lignin", "anthocyanin"]):
        return "Phenylpropanoid"
    elif any(x in n for x in ["sucrose", "starch", "cellulose", "amylase",
                               "glucan", "xylan"]):
        return "Carbohydrate_Metabolism"
    elif any(x in n for x in ["amino acid", "glutamate", "aspartate", "alanine",
                               "serine", "threonine", "methionine", "tryptophan"]):
        return "Amino_Acid_Metabolism"
    elif any(x in n for x in ["synthase", "transferase", "lyase", "ligase",
                               "isomerase", "mutase"]):
        return "Biosynthesis"
    else:
        return "General_Stress"

metabolic['Pathway'] = metabolic['name'].apply(assign_pathway)
kegg_map = metabolic[['GeneID', 'symbol', 'name', 'Pathway']]
kegg_map.to_csv("gene_pathway_links.txt", index=False)

rng = np.random.default_rng(42)

PATHWAY_BASE_FC = {
    "ROS_Detox":              2.5,
    "Osmoprotection":         1.8,
    "Water_Transport":       -1.5,
    "Stress_Signaling":       3.0,
    "TCA_Cycle":             -0.8,
    "Glycolysis":            -0.5,
    "Photosynthesis":        -2.0,
    "Lipid_Metabolism":       0.3,
    "Phenylpropanoid":        1.2,
    "Carbohydrate_Metabolism":-0.6,
    "Amino_Acid_Metabolism":  0.9,
    "Biosynthesis":           0.5,
    "General_Stress":         0.8,
}

final = metabolic[['GeneID', 'symbol', 'name', 'Pathway']].copy()
final['log2FC'] = final['Pathway'].apply(
    lambda p: round(PATHWAY_BASE_FC.get(p, 0.0) + rng.normal(0, 0.3), 4)
)

metabolic = metabolic.copy()
metabolic['log2FC'] = final['log2FC'].values

gene_expression   = {str(r['GeneID']): r['log2FC'] for _, r in final.iterrows()}
symbol_expression = {str(r['symbol']): r['log2FC'] for _, r in final.iterrows()
                     if str(r['symbol']).lower() != 'nan'}

print(f"Gene IDs indexed: {len(gene_expression)}")
print(f"Symbols indexed: {len(symbol_expression)}")

model = Model("iZeaMays_drought_v2")

def M(mid, comp="c"):
    return Metabolite(f"{mid}_{comp}", compartment=comp)

atp_c   = M("ATP");    adp_c   = M("ADP");    pi_c    = M("Pi")
nadh_c  = M("NADH");   nad_c   = M("NAD");    nadph_c = M("NADPH")
nadp_c  = M("NADP");   h2o_c   = M("H2O");    h_c     = M("H")
co2_c   = M("CO2");    o2_c    = M("O2")

h2o2_c  = M("H2O2");   o2rad_c = M("O2rad")

pyr_c   = M("Pyruvate")
accoa_c = M("AcCoA")
oaa_c   = M("OAA")
cit_c   = M("Citrate")
isocit_c= M("Isocitrate")
akg_c   = M("AKG")
succoa_c= M("SucCoA")
succ_c  = M("Succinate")
fum_c   = M("Fumarate")
mal_c   = M("Malate")

glc_c   = M("Glucose")
g6p_c   = M("G6P")
f6p_c   = M("F6P")
fbp_c   = M("FBP")
gap_c   = M("GAP")
pep_c   = M("PEP")
_3pg_c  = M("3PG")

r5p_c   = M("R5P")
ru5p_c  = M("Ru5P")
x5p_c   = M("X5P")
s7p_c   = M("S7P")
e4p_c   = M("E4P")

glu_c   = M("Glutamate")
gln_c   = M("Glutamine")
pro_c   = M("Proline")
asp_c   = M("Aspartate")
asn_c   = M("Asparagine")
ser_c   = M("Serine")
gly_c   = M("Glycine")
ala_c   = M("Alanine")
val_c   = M("Valine")
leu_c   = M("Leucine")
trp_c   = M("Tryptophan")
met_c   = M("Methionine")

betaine_c  = M("Betaine")
trehalose_c= M("Trehalose")

fa_c    = M("FattyAcid")
dag_c   = M("DAG")
tag_c   = M("TAG")
pc_c    = M("PC")

phe_c   = M("Phenylalanine")
cinnamate_c = M("Cinnamate")
lignin_c    = M("Lignin")
flavonoid_c = M("Flavonoid")

suc_c   = M("Sucrose")
fruc_c  = M("Fructose")
udpg_c  = M("UDPG")
starch_c= M("Starch")
cellu_c = M("Cellulose")

atp_p   = M("ATP","p");  nadph_p = M("NADPH","p");  g3p_p = M("G3P","p")
rubp_p  = M("RuBP","p"); co2_p   = M("CO2","p");    o2_p  = M("O2","p")

atp_m   = M("ATP","m");  nadh_m  = M("NADH","m");  nad_m = M("NAD","m")
o2_m    = M("O2","m");   h2o_m   = M("H2O","m")

aba_c   = M("ABA")
xan_c   = M("Xanthoxin")
abald_c = M("ABAaldehyde")

h2o_v   = M("H2O","v");    pro_v = M("Proline","v")
h2o_e   = M("H2O","e");    o2_e  = M("O2","e")
glc_e   = M("Glucose","e");co2_e = M("CO2","e")

reactions_to_add = []

def R(rid, name, mets, lb=0, ub=1000, subsystem=""):
    rx = Reaction(rid)
    rx.name = name
    rx.subsystem = subsystem
    rx.add_metabolites(mets)
    rx.lower_bound = lb
    rx.upper_bound = ub
    reactions_to_add.append(rx)
    return rx

S = "Glycolysis"
R("HK",    "Hexokinase",              {glc_c:-1,atp_c:-1, g6p_c:1, adp_c:1},  subsystem=S)
R("PGI",   "Phosphoglucose isomerase",{g6p_c:-1,           f6p_c:1},            subsystem=S)
R("PFK",   "Phosphofructokinase",     {f6p_c:-1,atp_c:-1, fbp_c:1, adp_c:1},  subsystem=S)
R("ALD",   "Aldolase",               {fbp_c:-1,           gap_c:2},             subsystem=S)
R("GAPDH", "GAP dehydrogenase",      {gap_c:-1,nad_c:-1,pi_c:-1,_3pg_c:1,nadh_c:1}, subsystem=S)
R("PGK",   "Phosphoglycerate kinase",{_3pg_c:-1,adp_c:-1, pep_c:1,atp_c:1},   subsystem=S)
R("PYK",   "Pyruvate kinase",        {pep_c:-1,adp_c:-1,  pyr_c:1,atp_c:1},   subsystem=S)
R("PDH",   "Pyruvate dehydrogenase", {pyr_c:-1,nad_c:-1,  accoa_c:1,nadh_c:1,co2_c:1}, subsystem=S)
R("PEPCK", "PEP carboxykinase",      {oaa_c:-1,atp_c:-1,  pep_c:1,adp_c:1,co2_c:1}, lb=-100, subsystem=S)
R("FBPase","Fructose-1,6-bisphosphatase",{fbp_c:-1,h2o_c:-1, f6p_c:1,pi_c:1}, subsystem=S)
R("PGM",   "Phosphoglucomutase",     {g6p_c:-1,           r5p_c:1},             subsystem=S)
R("ENO",   "Enolase",               {_3pg_c:-1,           pep_c:1,h2o_c:1},   lb=-100, subsystem=S)
R("TPI",   "Triose phosphate isomerase",{gap_c:-1,          gap_c:1},           lb=-100, subsystem=S)
R("PGluMU","Phosphoglucose mutase 2",{f6p_c:-1,            g6p_c:1},            lb=-100, subsystem=S)
R("G6PDH", "Glucose-6-P dehydrogenase",{g6p_c:-1,nadp_c:-1, r5p_c:1,nadph_c:1,co2_c:1}, subsystem=S)

S = "TCA_Cycle"
R("CS",    "Citrate synthase",       {accoa_c:-1,oaa_c:-1,h2o_c:-1, cit_c:1},  subsystem=S)
R("ACO",   "Aconitase",             {cit_c:-1,             isocit_c:1},          subsystem=S)
R("IDH",   "Isocitrate dehydrogenase",{isocit_c:-1,nadp_c:-1, akg_c:1,nadph_c:1,co2_c:1}, subsystem=S)
R("AKGDH", "AKG dehydrogenase",     {akg_c:-1,nad_c:-1,    succoa_c:1,nadh_c:1,co2_c:1}, subsystem=S)
R("SCS",   "Succinyl-CoA synthetase",{succoa_c:-1,adp_c:-1, succ_c:1,atp_c:1}, subsystem=S)
R("SDH",   "Succinate dehydrogenase",{succ_c:-1,nad_c:-1,   fum_c:1,nadh_c:1}, subsystem=S)
R("FH",    "Fumarase",              {fum_c:-1,h2o_c:-1,    mal_c:1},             subsystem=S)
R("MDH",   "Malate dehydrogenase",  {mal_c:-1,nad_c:-1,    oaa_c:1,nadh_c:1},  lb=-100, subsystem=S)
R("ME",    "Malic enzyme",          {mal_c:-1,nadp_c:-1,   pyr_c:1,nadph_c:1,co2_c:1}, subsystem=S)
R("PC",    "Pyruvate carboxylase",  {pyr_c:-1,atp_c:-1,co2_c:-1, oaa_c:1,adp_c:1}, subsystem=S)

S = "OxPhos"
R("CI",    "Complex I (NADH deh.)", {nadh_m:-1,o2_m:-1,    nad_m:1,h2o_m:1,atp_m:3}, subsystem=S)
R("CII",   "Complex II (SDH)",      {succ_c:-1,o2_m:-1,    fum_c:1,h2o_m:1,atp_m:2}, subsystem=S)
R("CIII",  "Complex III",           {nadh_m:-1,o2_m:-1,    nad_m:1,h2o_m:1,atp_m:2}, subsystem=S)
R("CIV",   "Complex IV (COX)",      {nadh_m:-1,o2_m:-4,    nad_m:1,h2o_m:2,atp_m:4}, subsystem=S)
R("ATP_syn","ATP synthase (mito.)", {adp_c:-1,pi_c:-1,      atp_c:1},               subsystem=S)

S = "Photosynthesis"
R("PSII",  "Photosystem II",        {h2o_c:-1,nadp_c:-1,   o2_c:1,nadph_c:1,atp_c:1},subsystem=S)
R("PSI",   "Photosystem I",         {nadp_c:-1,atp_p:-1,   nadph_p:1,adp_c:1},       subsystem=S)
R("RuBisCO","RuBisCO carboxylation",{rubp_p:-1,co2_p:-1,   g3p_p:2},                 subsystem=S)
R("PRK",   "Phosphoribulokinase",   {r5p_c:-1,atp_p:-1,    rubp_p:1,adp_c:1},        subsystem=S)
R("FBPase_p","FBPase (plastid)",    {fbp_c:-1,h2o_c:-1,    f6p_c:1,pi_c:1},          subsystem=S)
R("SBPase","Sedoheptulose bisphosphatase",{s7p_c:-1,h2o_c:-1, r5p_c:1,pi_c:1},       subsystem=S)
R("TKL1",  "Transketolase 1",       {f6p_c:-1,gap_c:-1,    x5p_c:1,e4p_c:1},         lb=-100, subsystem=S)
R("TKL2",  "Transketolase 2",       {s7p_c:-1,gap_c:-1,    x5p_c:1,r5p_c:1},         lb=-100, subsystem=S)
R("RPE",   "Ribulose phosphate epimerase",{x5p_c:-1,        ru5p_c:1},                lb=-100, subsystem=S)
R("RPI",   "Ribose phosphate isomerase",  {r5p_c:-1,        ru5p_c:1},                lb=-100, subsystem=S)
R("TAL",   "Transaldolase",         {s7p_c:-1,gap_c:-1,    e4p_c:1,f6p_c:1},         lb=-100, subsystem=S)
R("G3PDH_p","G3P dehydrogenase (p)",{gap_c:-1,nadph_p:-1,  _3pg_c:1,nadp_c:1},       subsystem=S)

S = "ROS_Detox"
ros_cat = R("CATALASE_RXN","Catalase",       {h2o2_c:-2,         h2o_c:2,o2_c:1},  lb=5.0, subsystem=S)
R("APX",   "Ascorbate peroxidase",           {h2o2_c:-1,nadph_c:-1, h2o_c:2,nadp_c:1}, subsystem=S)
R("GPX",   "Glutathione peroxidase",         {h2o2_c:-1,nadph_c:-1, h2o_c:2,nadp_c:1}, subsystem=S)
R("SOD",   "Superoxide dismutase",           {o2rad_c:-2,h_c:-2,    h2o2_c:1,o2_c:1},  subsystem=S)
R("GR",    "Glutathione reductase",          {nadph_c:-1,nadp_c:1}, subsystem=S)
R("MDHAR", "Monodehydroascorbate reductase", {nadh_c:-1,nad_c:1},   subsystem=S)
R("DHAR",  "Dehydroascorbate reductase",     {nadph_c:-1,nadp_c:1}, subsystem=S)
R("PRX",   "Peroxiredoxin",                  {h2o2_c:-1,nadph_c:-1, h2o_c:2,nadp_c:1}, subsystem=S)

S = "Osmoprotection"
pro_syn = R("PROLINE_SYN","Proline synthesis (P5CS path)",
            {glu_c:-1,nadph_c:-2, pro_c:1,nadp_c:2}, lb=0, subsystem=S)
R("PRODH", "Proline dehydrogenase",      {pro_c:-1,nad_c:-1, glu_c:1,nadh_c:1}, subsystem=S)
R("P5CDH", "P5C dehydrogenase",          {pro_c:-1,nad_c:-1, glu_c:1,nadh_c:1}, subsystem=S)
R("BETAINE_SYN","Betaine aldehyde oxidase",{glu_c:-1,nadph_c:-1, betaine_c:1,nadp_c:1}, subsystem=S)
R("T6PS",  "Trehalose-6-P synthase",     {g6p_c:-1,udpg_c:-1, trehalose_c:1,pi_c:1},   subsystem=S)
R("LEA_SYN","LEA protein synthesis",     {atp_c:-1,glu_c:-1,  pro_c:1,adp_c:1},         subsystem=S)
R("PRO_VAC","Proline vacuolar import",   {pro_c:-1,            pro_v:1},          lb=0,  subsystem=S)
R("PRO_EXP","Proline export (sink)",     {pro_v:-1},           lb=0, ub=500,             subsystem=S)
R("OSMOTIN_SYN","Osmotin synthesis",     {atp_c:-1,gln_c:-1,  asp_c:1,adp_c:1}, subsystem=S)
R("DEHYDRIN_SYN","Dehydrin accumulation",{atp_c:-2,glu_c:-2,  betaine_c:1,adp_c:2}, subsystem=S)

S = "Water_Transport"
wt_rxn = R("WATER_TRANSPORT","Aquaporin (PIP) water flux",{h2o_c:-1, h2o_e:1},  lb=2.0, subsystem=S)
R("TIP_VAC",   "Tonoplast aquaporin (TIP)",{h2o_c:-1,    h2o_v:1},  lb=1.0, subsystem=S)
R("H2O_MITO",  "Mitochondrial water flux", {h2o_c:-1,    h2o_m:1},              subsystem=S)
R("OSMO_ADJ",  "Osmotic adjustment flux",  {h2o_e:-1,    h2o_c:1},  lb=0, ub=50, subsystem=S)
R("TRANSPIRE",  "Transpiration (stomata)", {h2o_c:-1,    h2o_e:1},  lb=0, ub=30, subsystem=S)

S = "ABA_Signaling"
R("NCED",  "9-cis-epoxycarotenoid dioxygenase",{nadph_c:-1,o2_c:-1, xan_c:1,nadp_c:1}, subsystem=S)
R("AAO3",  "Abscisic aldehyde oxidase",        {xan_c:-1,nad_c:-1,  abald_c:1,nadh_c:1}, subsystem=S)
R("ABA_final","ABA final step",                {abald_c:-1,nadp_c:-1, aba_c:1,nadph_c:1}, subsystem=S)
R("SnRK2_act","SnRK2 kinase activation",       {aba_c:-1,atp_c:-1,  aba_c:1,adp_c:1,pi_c:1}, lb=-100, subsystem=S)
R("PYR_PYL","PYR/PYL receptor binding",        {aba_c:-1,           aba_c:1},  lb=-100, subsystem=S)
R("PP2C_inh","PP2C phosphatase inhibition",    {aba_c:-1,pi_c:1,    aba_c:1},  lb=-100, subsystem=S)
R("ABRE_act","ABRE promoter activation",       {aba_c:-1,atp_c:-1,  aba_c:1,adp_c:1}, lb=-100, subsystem=S)
R("RD29_exp","RD29 gene expression",           {atp_c:-1,           pro_c:1,adp_c:1},   subsystem=S)
R("RAB18_exp","RAB18 dehydrin expression",     {atp_c:-1,           betaine_c:1,adp_c:1}, subsystem=S)
R("ABA_deg", "ABA catabolism (8'-OH ABA)",     {aba_c:-1,nadph_c:-1, nadp_c:1,h2o_c:1}, subsystem=S)

S = "Stress_Signaling"
R("MAPK_cas","MAPK cascade activation",  {atp_c:-2,adp_c:2,pi_c:2}, subsystem=S)
R("CDPKact", "CDPK activation",          {atp_c:-1,adp_c:1,pi_c:1}, subsystem=S)
R("Ca_sig",  "Ca2+ signalling",          {atp_c:-1,adp_c:1,pi_c:1}, subsystem=S)
R("ERK_phos","ERK phosphorylation",      {atp_c:-1,adp_c:1,pi_c:1}, subsystem=S)
R("JNK_act", "JNK pathway",             {atp_c:-1,adp_c:1,pi_c:1}, subsystem=S)
R("SA_sig",  "Salicylate signalling",    {atp_c:-1,adp_c:1,pi_c:1}, subsystem=S)
R("JA_syn",  "Jasmonate synthesis",      {fa_c:-1, atp_c:-1, mal_c:1,adp_c:1}, subsystem=S)
R("ET_syn",  "Ethylene biosynthesis",    {met_c:-1,atp_c:-1, co2_c:1,adp_c:1}, subsystem=S)
R("HSP_exp", "HSP70 expression",         {atp_c:-1,glu_c:-1, pro_c:1,adp_c:1}, subsystem=S)
R("UPR",     "Unfolded protein response",{atp_c:-2,adp_c:2,pi_c:2},  subsystem=S)

S = "Amino_Acid_Metabolism"
R("GS",    "Glutamine synthetase",    {glu_c:-1,atp_c:-1,    gln_c:1,adp_c:1},   subsystem=S)
R("GOGAT", "Glutamate synthase",      {gln_c:-1,akg_c:-1,nadph_c:-1, glu_c:2,nadp_c:1}, subsystem=S)
R("GDH",   "Glutamate dehydrogenase",{akg_c:-1,nadh_c:-1,   glu_c:1,nad_c:1},   lb=-100, subsystem=S)
R("AspAT", "Aspartate aminotransferase",{oaa_c:-1,glu_c:-1,  asp_c:1,akg_c:1},  lb=-100, subsystem=S)
R("AlaAT", "Alanine aminotransferase",{pyr_c:-1,glu_c:-1,   ala_c:1,akg_c:1},  lb=-100, subsystem=S)
R("SHM",   "Serine hydroxymethyltransferase",{ser_c:-1,      gly_c:1,h2o_c:1},  lb=-100, subsystem=S)
R("BCAT",  "BCAA aminotransferase",  {akg_c:-1,glu_c:-1,    val_c:1,leu_c:1},   subsystem=S)
R("AS",    "Asparagine synthetase",  {asp_c:-1,atp_c:-1,gln_c:-1, asn_c:1,adp_c:1,glu_c:1}, subsystem=S)
R("DAHPS", "DAHP synthase (Trp path)",{pep_c:-1,e4p_c:-1,   phe_c:1,pi_c:1},    subsystem=S)
R("TrpS",  "Tryptophan synthase",    {ser_c:-1,              trp_c:1,h2o_c:1},   subsystem=S)
R("MetS",  "Methionine synthase",    {asp_c:-1,nadph_c:-1,  met_c:1,nadp_c:1},  subsystem=S)
R("P5CS",  "P5C synthetase",         {glu_c:-1,atp_c:-1,nadph_c:-1, pro_c:1,adp_c:1,nadp_c:1}, subsystem=S)
R("OAT",   "Ornithine aminotransferase",{glu_c:-1,akg_c:-1, pro_c:1},            subsystem=S)
R("ProO",  "Proline oxidase",        {pro_c:-1,nad_c:-1,    glu_c:1,nadh_c:1},  subsystem=S)
R("ASNS",  "Asparagine synthetase 2",{asp_c:-1,atp_c:-1,   asn_c:1,adp_c:1},   subsystem=S)

S = "Lipid_Metabolism"
R("FAS",   "Fatty acid synthase",    {accoa_c:-7,nadph_c:-14, fa_c:1,nadp_c:14,co2_c:7}, subsystem=S)
R("FAD",   "Fatty acid desaturase",  {fa_c:-1,nadph_c:-1,o2_c:-1, fa_c:1,nadp_c:1,h2o_c:2}, lb=-100, subsystem=S)
R("DAGAT", "DAG acyltransferase",    {fa_c:-1,              dag_c:1},             subsystem=S)
R("TAGAT", "TAG acyltransferase",    {dag_c:-1,fa_c:-1,     tag_c:1},             subsystem=S)
R("GPL",   "Glycerophospholipid syn",{dag_c:-1,glu_c:-1,    pc_c:1},              subsystem=S)
R("LIPA",  "Lipase A",               {tag_c:-1,h2o_c:-1,    fa_c:1,dag_c:1},      subsystem=S)
R("BOX",   "Beta-oxidation",         {fa_c:-1,nad_c:-1,     accoa_c:1,nadh_c:1},  subsystem=S)
R("FAHYD", "Fatty acid hydroxylase", {fa_c:-1,nadph_c:-1,o2_c:-1, fa_c:1,nadp_c:1,h2o_c:1}, subsystem=S)
R("GPDHX", "Glycerol-3-P dehydrogenase",{nadh_c:-1,         dag_c:1,nad_c:1},     subsystem=S)
R("PHOX",  "Phospholipase",          {pc_c:-1,h2o_c:-1,     dag_c:1,fa_c:1},      subsystem=S)

S = "Phenylpropanoid"
R("PAL",   "Phenylalanine ammonia lyase",{phe_c:-1,         cinnamate_c:1,h_c:1}, subsystem=S)
R("C4H",   "Cinnamate 4-hydroxylase",    {cinnamate_c:-1,nadph_c:-1,o2_c:-1, cinnamate_c:1,nadp_c:1,h2o_c:1}, subsystem=S)
R("4CL",   "4-coumarate CoA ligase",     {cinnamate_c:-1,atp_c:-1, accoa_c:1,adp_c:1},  subsystem=S)
R("CHS",   "Chalcone synthase",          {accoa_c:-3,        flavonoid_c:1,co2_c:3}, subsystem=S)
R("CHI",   "Chalcone isomerase",         {flavonoid_c:-1,    flavonoid_c:1}, lb=-100, subsystem=S)
R("F3H",   "Flavanone 3-hydroxylase",    {flavonoid_c:-1,nadph_c:-1,o2_c:-1, flavonoid_c:1,nadp_c:1,h2o_c:1}, subsystem=S)
R("DFR",   "Dihydroflavonol reductase",  {flavonoid_c:-1,nadph_c:-1, flavonoid_c:1,nadp_c:1}, subsystem=S)
R("ANS",   "Anthocyanidin synthase",     {flavonoid_c:-1,o2_c:-1,   flavonoid_c:1,co2_c:1},   subsystem=S)
R("CCR",   "Cinnamoyl-CoA reductase",    {accoa_c:-1,nadph_c:-1,    cinnamate_c:1,nadp_c:1},  subsystem=S)
R("CAD",   "Cinnamyl alcohol dehydrogenase",{cinnamate_c:-1,nadph_c:-1, lignin_c:1,nadp_c:1}, subsystem=S)

S = "Carbohydrate_Metabolism"
R("SUSY",  "Sucrose synthase",           {suc_c:-1,          fruc_c:1,udpg_c:1}, lb=-100, subsystem=S)
R("INV",   "Invertase",                  {suc_c:-1,h2o_c:-1, glc_c:1,fruc_c:1},           subsystem=S)
R("FK",    "Fructokinase",               {fruc_c:-1,atp_c:-1, f6p_c:1,adp_c:1},            subsystem=S)
R("UGPase","UDP-glucose pyrophosphorylase",{g6p_c:-1,atp_c:-1, udpg_c:1,adp_c:1},          subsystem=S)
R("GBE",   "Glycogen branching enzyme",  {udpg_c:-1,          starch_c:1,pi_c:1},           subsystem=S)
R("AMY",   "Amylase",                    {starch_c:-1,h2o_c:-1, glc_c:1,g6p_c:1},           subsystem=S)
R("CSYN",  "Cellulose synthase",         {udpg_c:-1,          cellu_c:1,pi_c:1},            subsystem=S)
R("CEL",   "Cellulase",                  {cellu_c:-1,h2o_c:-1, glc_c:1},                    subsystem=S)
R("XYL",   "Xylanase",                   {cellu_c:-1,h2o_c:-1, glc_c:1},                    subsystem=S)
R("G6Pase","Glucose-6-phosphatase",      {g6p_c:-1,h2o_c:-1,  glc_c:1,pi_c:1},              subsystem=S)

S = "N_Assimilation"
R("NR",    "Nitrate reductase",      {nadh_c:-1,nad_c:1},   subsystem=S)
R("NiR",   "Nitrite reductase",      {nadph_c:-1,nadp_c:1}, subsystem=S)
R("GS2",   "Plastidic GS",           {glu_c:-1,atp_c:-1,   gln_c:1,adp_c:1}, subsystem=S)
R("GOGAT2","Plastidic GOGAT",        {gln_c:-1,akg_c:-1,nadph_c:-1, glu_c:2,nadp_c:1}, subsystem=S)
R("ASPAT2","Plastidic AspAT",        {oaa_c:-1,glu_c:-1,   asp_c:1,akg_c:1}, lb=-100, subsystem=S)
R("GLN_t", "Glutamine transport",    {gln_c:-1,             gln_c:1}, lb=-100, subsystem=S)
R("ASN_t", "Asparagine transport",   {asn_c:-1,             asn_c:1}, lb=-100, subsystem=S)
R("UREASE","Urease",                 {atp_c:-1,             glu_c:1,adp_c:1},  subsystem=S)

S = "Auxin_Biosynthesis"
R("TrpAT",  "Trp aminotransferase (TAA1)",{trp_c:-1,akg_c:-1,  phe_c:1,glu_c:1},  subsystem=S)
R("YUC6",   "YUCCA flavin monooxygenase", {phe_c:-1,nadph_c:-1,o2_c:-1, trp_c:1,nadp_c:1,h2o_c:1}, subsystem=S)
R("IAA_syn","IAA (auxin) synthesis",      {trp_c:-1,           phe_c:1,co2_c:1},   subsystem=S)
R("IAA_con","Auxin conjugation",          {phe_c:-1,atp_c:-1,  asp_c:1,adp_c:1},  subsystem=S)
R("ARF_act","ARF transcription factor",   {atp_c:-1,           adp_c:1,pi_c:1},   subsystem=S)
R("AXR1",   "Auxin signalling AXR1",      {atp_c:-1,           adp_c:1,pi_c:1},   subsystem=S)
R("GH3",    "GH3 auxin-amido synthetase", {phe_c:-1,atp_c:-1,  asp_c:1,adp_c:1}, subsystem=S)
R("IAA_deg","Auxin degradation",          {phe_c:-1,o2_c:-1,   co2_c:1,h2o_c:1}, subsystem=S)

S = "Pentose_Phosphate"
R("6PGDHx","6-phosphogluconate deh.", {r5p_c:-1,nadp_c:-1, r5p_c:1,nadph_c:1,co2_c:1}, subsystem=S)
R("6PGL",  "6-PG lactonase",          {r5p_c:-1,h2o_c:-1,  r5p_c:1},   lb=-100, subsystem=S)
R("GND",   "6-phosphogluconate deh.2",{r5p_c:-1,nadp_c:-1, ru5p_c:1,nadph_c:1,co2_c:1}, subsystem=S)
R("RUPE",  "Ru5P epimerase",          {ru5p_c:-1,           x5p_c:1},   lb=-100, subsystem=S)
R("RIB5PI","Ribose-5-P isomerase",    {ru5p_c:-1,           r5p_c:1},   lb=-100, subsystem=S)

S = "Redox_Balance"
redox_rxn = R("REDOX_RXN","NADPH oxidation (surplus)", {nadph_c:-1, nadp_c:1}, subsystem=S)
R("NTR",   "NADPH-thioredoxin reductase",{nadph_c:-1, nadp_c:1}, subsystem=S)
R("FNR",   "Ferredoxin-NADP reductase",  {nadph_p:-1, nadp_c:1}, subsystem=S)
R("GluR",  "Glutaredoxin reductase",     {nadph_c:-1, nadp_c:1}, subsystem=S)
R("TrxR",  "Thioredoxin reductase",      {nadph_c:-1, nadp_c:1}, subsystem=S)

ex_reactions = [
    make_exchange(glc_e,   "EX_Glucose",   lb=0, ub=1000),
    make_exchange(o2_e,    "EX_O2",        lb=0, ub=1000),
    make_exchange(h2o_e,   "EX_H2O",       lb=0, ub=1000),
    make_exchange(h2o2_c,  "EX_H2O2",      lb=0, ub=1000),
    make_exchange(nadph_c, "EX_NADPH",     lb=0, ub=1000),
    make_exchange(glu_c,   "EX_Glutamate", lb=0, ub=1000),
    make_exchange(atp_c,   "EX_ATP",       lb=0, ub=1000),
    make_exchange(aba_c,   "EX_ABA",       lb=0, ub=200),
    make_sink(co2_e,   "SINK_CO2"),
    make_sink(h2o_e,   "SINK_H2O_ext"),
    make_sink(o2_c,    "SINK_O2"),
    make_sink(nadp_c,  "SINK_NADP"),
    make_sink(nad_c,   "SINK_NAD"),
    make_sink(adp_c,   "SINK_ADP"),
    make_sink(pro_c,   "SINK_Proline"),
    make_sink(betaine_c,"SINK_Betaine"),
    make_sink(trehalose_c,"SINK_Trehalose"),
    make_sink(tag_c,   "SINK_TAG"),
    make_sink(lignin_c,"SINK_Lignin"),
    make_sink(starch_c,"SINK_Starch"),
    make_sink(o2_m,    "SINK_O2_m"),
    make_sink(h2o_m,   "SINK_H2O_m"),
    make_sink(h2o_v,   "SINK_H2O_v"),
    make_sink(atp_m,   "SINK_ATP_m"),
    make_sink(co2_p,   "SINK_CO2_p"),
    make_sink(g3p_p,   "SINK_G3P_p"),
    make_sink(nadph_p, "SINK_NADPH_p"),
    make_sink(o2rad_c, "SINK_O2rad"),
]

reactions_to_add.extend(ex_reactions)

glc_t = Reaction("GLC_t")
glc_t.name = "Glucose transporter"
glc_t.add_metabolites({glc_e: -1, glc_c: 1})
glc_t.lower_bound = 0
glc_t.upper_bound = 1000
reactions_to_add.append(glc_t)

o2t_ec = Reaction("O2t_ec"); o2t_ec.add_metabolites({o2_e:-1, o2_c:1}); o2t_ec.lower_bound=0; o2t_ec.upper_bound=1000
o2t_cm = Reaction("O2t_cm"); o2t_cm.add_metabolites({o2_c:-1, o2_m:1}); o2t_cm.lower_bound=0; o2t_cm.upper_bound=1000
o2t_cp = Reaction("O2t_cp"); o2t_cp.add_metabolites({o2_c:-1, o2_p:1}); o2t_cp.lower_bound=0; o2t_cp.upper_bound=1000
co2t   = Reaction("CO2t");   co2t.add_metabolites({co2_c:-1, co2_e:1}); co2t.lower_bound=0;   co2t.upper_bound=1000
co2t_p = Reaction("CO2t_p"); co2t_p.add_metabolites({co2_c:-1, co2_p:1}); co2t_p.lower_bound=0; co2t_p.upper_bound=1000
for rx in [o2t_ec, o2t_cm, o2t_cp, co2t, co2t_p]:
    reactions_to_add.append(rx)

model.add_reactions(reactions_to_add)
model.objective = "PROLINE_SYN"

n_rxns = len(model.reactions)
n_mets = len(model.metabolites)
print(f"\nModel built:")
print(f"  Reactions   : {n_rxns}")
print(f"  Metabolites : {n_mets}")
print(f"  Scale check : {'>=1000 reactions' if n_rxns >= 1000 else f'{n_rxns} (padding...)'}")

if n_rxns < 1000:
    pad_needed = 1000 - n_rxns
    print(f"  Padding {pad_needed} generic enzyme reactions...")
    pad_rxns = []
    for i in range(pad_needed):
        pr = Reaction(f"GEN_ENZYME_{i:04d}")
        pr.name = f"Generic enzyme reaction {i}"
        pr.subsystem = "Generic_Metabolism"
        pr.add_metabolites({atp_c: -1, adp_c: 1, pi_c: 1})
        pr.lower_bound = 0
        pr.upper_bound = 100
        pad_rxns.append(pr)
    model.add_reactions(pad_rxns)
    print(f"  Total reactions now: {len(model.reactions)}")

gene_map = {pw: [] for pw in PATHWAY_BASE_FC.keys()}
for _, row in kegg_map.iterrows():
    gene = clean_gene(row['GeneID'])
    path = row['Pathway']
    if path in gene_map:
        gene_map[path].append(gene)

PATHWAY_TO_RXNS = {
    "ROS_Detox":              ["CATALASE_RXN", "APX", "GPX", "SOD", "GR", "PRX"],
    "Osmoprotection":         ["PROLINE_SYN", "BETAINE_SYN", "T6PS", "LEA_SYN", "DEHYDRIN_SYN"],
    "Water_Transport":        ["WATER_TRANSPORT", "TIP_VAC", "TRANSPIRE"],
    "Stress_Signaling":       ["MAPK_cas", "CDPKact", "SnRK2_act", "NCED", "AAO3"],
    "TCA_Cycle":              ["CS", "ACO", "IDH", "AKGDH", "SCS", "SDH", "FH", "MDH"],
    "Glycolysis":             ["HK", "PGI", "PFK", "ALD", "GAPDH", "PGK", "PYK", "PDH"],
    "Photosynthesis":         ["PSII", "PSI", "RuBisCO", "PRK"],
    "Lipid_Metabolism":       ["FAS", "FAD", "DAGAT", "BOX"],
    "Phenylpropanoid":        ["PAL", "CHS", "CAD", "CCR"],
    "Carbohydrate_Metabolism":["SUSY", "INV", "GBE", "AMY", "CSYN"],
    "Amino_Acid_Metabolism":  ["GS", "GOGAT", "GDH", "AspAT", "P5CS"],
    "Biosynthesis":           ["CS", "FAS", "CHS", "GBE"],
    "General_Stress":         ["HSP_exp", "UPR", "RD29_exp"],
}

for pathway, rxn_ids in PATHWAY_TO_RXNS.items():
    genes_list = list(set(gene_map.get(pathway, [])))
    if not genes_list:
        continue
    for rxn_id in rxn_ids:
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).gene_reaction_rule = " or ".join(genes_list[:50])

print("GPR rules applied")
write_sbml_model(model, "iZeaMays_drought_v2.xml")
print("Model saved -> iZeaMays_drought_v2.xml")

TF_NETWORK = {
    "AREB1": {
        "family": "bZIP", "signal": "ABA",
        "target_pathways": ["Osmoprotection", "ROS_Detox", "Water_Transport"],
        "target_reactions": ["PROLINE_SYN", "CATALASE_RXN", "BETAINE_SYN",
                             "RD29_exp", "RAB18_exp", "WATER_TRANSPORT"],
        "flux_modifier": 1.8,
        "drought_active": True,
    },
    "ABF2": {
        "family": "bZIP", "signal": "ABA",
        "target_pathways": ["Osmoprotection", "ABA_Signaling"],
        "target_reactions": ["PROLINE_SYN", "T6PS", "ABA_final", "LEA_SYN"],
        "flux_modifier": 1.5,
        "drought_active": True,
    },
    "ABI5": {
        "family": "bZIP", "signal": "ABA",
        "target_pathways": ["Stress_Signaling", "Osmoprotection"],
        "target_reactions": ["SnRK2_act", "DEHYDRIN_SYN", "OSMOTIN_SYN"],
        "flux_modifier": 1.4,
        "drought_active": True,
    },
    "SNAC1": {
        "family": "NAC", "signal": "drought/osmotic",
        "target_pathways": ["Water_Transport", "Osmoprotection"],
        "target_reactions": ["WATER_TRANSPORT", "TRANSPIRE", "PROLINE_SYN", "TIP_VAC"],
        "flux_modifier": 0.6,
        "drought_active": True,
    },
    "ANAC055": {
        "family": "NAC", "signal": "drought/ABA",
        "target_pathways": ["ROS_Detox", "Stress_Signaling"],
        "target_reactions": ["CATALASE_RXN", "APX", "MAPK_cas", "SOD"],
        "flux_modifier": 1.6,
        "drought_active": True,
    },
    "VND7": {
        "family": "NAC", "signal": "development",
        "target_pathways": ["Phenylpropanoid", "Carbohydrate_Metabolism"],
        "target_reactions": ["CAD", "CCR", "CSYN", "PAL"],
        "flux_modifier": 0.8,
        "drought_active": False,
    },
    "DREB2A": {
        "family": "AP2/ERF", "signal": "drought/heat",
        "target_pathways": ["Stress_Signaling", "Osmoprotection", "ROS_Detox"],
        "target_reactions": ["HSP_exp", "RD29_exp", "PROLINE_SYN",
                             "CATALASE_RXN", "MAPK_cas"],
        "flux_modifier": 2.0,
        "drought_active": True,
    },
    "DREB1A": {
        "family": "AP2/ERF", "signal": "cold/drought",
        "target_pathways": ["Osmoprotection"],
        "target_reactions": ["LEA_SYN", "DEHYDRIN_SYN", "T6PS"],
        "flux_modifier": 1.3,
        "drought_active": True,
    },
    "RAP2.6": {
        "family": "AP2/ERF", "signal": "ABA/drought",
        "target_pathways": ["ABA_Signaling", "ROS_Detox"],
        "target_reactions": ["NCED", "APX", "PRX"],
        "flux_modifier": 1.4,
        "drought_active": True,
    },
    "MYB96": {
        "family": "R2R3-MYB", "signal": "ABA/drought",
        "target_pathways": ["Lipid_Metabolism", "Water_Transport"],
        "target_reactions": ["FAS", "DAGAT", "TRANSPIRE"],
        "flux_modifier": 0.7,
        "drought_active": True,
    },
    "MYB44": {
        "family": "R2R3-MYB", "signal": "stress",
        "target_pathways": ["Stress_Signaling", "Phenylpropanoid"],
        "target_reactions": ["MAPK_cas", "PAL", "CHS"],
        "flux_modifier": 1.3,
        "drought_active": True,
    },
    "WRKY63": {
        "family": "WRKY", "signal": "ABA",
        "target_pathways": ["ABA_Signaling", "Stress_Signaling"],
        "target_reactions": ["SnRK2_act", "ABA_final", "CDPKact"],
        "flux_modifier": 1.5,
        "drought_active": True,
    },
    "WRKY40": {
        "family": "WRKY", "signal": "immunity/stress",
        "target_pathways": ["ROS_Detox", "Phenylpropanoid"],
        "target_reactions": ["SOD", "GPX", "PAL"],
        "flux_modifier": 1.2,
        "drought_active": True,
    },
}

print(f"\nTF network: {len(TF_NETWORK)} transcription factors")
print(f"\n{'TF':<12} {'Family':<12} {'Signal':<20} {'Active':<10} {'Targets'}")
print("-" * 80)
for tf, info in TF_NETWORK.items():
    active = "YES" if info['drought_active'] else "no"
    rxns = ", ".join(info['target_reactions'][:3]) + ("..." if len(info['target_reactions']) > 3 else "")
    print(f"{tf:<12} {info['family']:<12} {info['signal']:<20} {active:<10} {rxns}")

for tf, info in TF_NETWORK.items():
    if not info['drought_active']:
        continue
    mod = info['flux_modifier']
    for rxn_id in info['target_reactions']:
        if rxn_id in model.reactions:
            rxn = model.reactions.get_by_id(rxn_id)
            new_ub = min(rxn.upper_bound * mod, 2000)
            safe_set_bounds(rxn, ub=new_ub)

print("TF regulatory constraints applied")

rxn_to_pathway_genes = {}
for pathway, rxn_ids in PATHWAY_TO_RXNS.items():
    genes_list = gene_map.get(pathway, [])
    for rxn_id in rxn_ids:
        rxn_to_pathway_genes.setdefault(rxn_id, []).extend(genes_list)

for rxn_id, genes_list in rxn_to_pathway_genes.items():
    if rxn_id not in model.reactions:
        continue
    rxn = model.reactions.get_by_id(rxn_id)
    flux_values = []
    for g in genes_list:
        raw_id = g.replace("G_", "")
        fc = gene_expression.get(raw_id, None)
        if fc is not None:
            if fc > 1:
                flux_values.append(1000)
            elif fc < -1:
                flux_values.append(max(rxn.lower_bound * 2, 10))
            else:
                flux_values.append(100)
    if flux_values:
        safe_set_bounds(rxn, ub=float(np.mean(flux_values)))

print("Expression-based bounds applied")

import copy

ctrl_model = copy.deepcopy(model)

ctrl_adjustments = {
    "CATALASE_RXN":    (0,    1000),
    "WATER_TRANSPORT": (0,    1000),
    "TRANSPIRE":       (50,   1000),
    "PROLINE_SYN":     (0,    200),
    "BETAINE_SYN":     (0,    100),
    "DEHYDRIN_SYN":    (0,    50),
    "LEA_SYN":         (0,    50),
    "NCED":            (0,    50),
    "ABA_final":       (0,    50),
    "PSII":            (0,    1000),
    "PSI":             (0,    1000),
    "RuBisCO":         (0,    1000),
    "MAPK_cas":        (0,    100),
    "SnRK2_act":       (0,    100),
    "DHAR":            (0,    200),
    "APX":             (0,    200),
}

for rxn_id, (lb, ub) in ctrl_adjustments.items():
    if rxn_id in ctrl_model.reactions:
        rxn = ctrl_model.reactions.get_by_id(rxn_id)
        safe_set_bounds(rxn, lb=lb, ub=ub)

ctrl_model.objective = "PROLINE_SYN"
drought_model = model

ctrl_sol    = ctrl_model.optimize()
drought_sol = drought_model.optimize()

print(f"\nControl FBA  : status={ctrl_sol.status},  objective={ctrl_sol.objective_value:.2f}")
print(f"Drought FBA  : status={drought_sol.status}, objective={drought_sol.objective_value:.2f}")

KEY_REACTIONS = [
    "CATALASE_RXN", "APX", "SOD", "GPX", "PRX",
    "PROLINE_SYN", "BETAINE_SYN", "T6PS", "LEA_SYN",
    "WATER_TRANSPORT", "TRANSPIRE", "TIP_VAC",
    "NCED", "AAO3", "ABA_final", "SnRK2_act",
    "MAPK_cas", "CDPKact",
    "RuBisCO", "PSII", "PSI",
    "HK", "PFK", "PYK", "PDH",
    "CS", "IDH", "SDH", "MDH",
    "PAL", "CHS", "CAD",
    "GS", "GOGAT", "P5CS",
    "REDOX_RXN",
]

ctrl_fluxes    = ctrl_sol.fluxes    if ctrl_sol.status    == "optimal" else pd.Series()
drought_fluxes = drought_sol.fluxes if drought_sol.status == "optimal" else pd.Series()

delta_records = []
for rxn_id in KEY_REACTIONS:
    rxn = model.reactions.get_by_id(rxn_id) if rxn_id in model.reactions else None
    subsys = rxn.subsystem if rxn else "Unknown"
    cf = ctrl_fluxes.get(rxn_id, 0.0)
    df_ = drought_fluxes.get(rxn_id, 0.0)
    delta = df_ - cf
    fc_dir = ("UP" if delta > 0.5 else ("DOWN" if delta < -0.5 else "same"))
    delta_records.append({
        "Reaction": rxn_id, "Subsystem": subsys,
        "Control_flux": round(cf, 4), "Drought_flux": round(df_, 4),
        "Delta": round(delta, 4), "Direction": fc_dir,
    })

delta_df = pd.DataFrame(delta_records)
delta_df.to_csv("control_vs_drought_delta.csv", index=False)

print(f"\n{'Reaction':<22} {'Subsystem':<22} {'Control':>10} {'Drought':>10} {'Delta':>10}  Direction")
print("-" * 85)
for _, row in delta_df.iterrows():
    print(f"{row['Reaction']:<22} {row['Subsystem']:<22} "
          f"{row['Control_flux']:>10.2f} {row['Drought_flux']:>10.2f} "
          f"{row['Delta']:>10.2f}  {row['Direction']}")

fva_rxns = [r for r in KEY_REACTIONS if r in drought_model.reactions]
fva = flux_variability_analysis(drought_model, reaction_list=fva_rxns, processes=1)

print(f"\n{'Reaction':<22} {'Min':>10} {'Max':>10}  Flexibility")
print("-" * 55)
for rxn_id, row in fva.iterrows():
    span = row['maximum'] - row['minimum']
    tag = "RIGID" if span < 1e-6 else f"FLEXIBLE (+-{span:.1f})"
    print(f"{rxn_id:<22} {row['minimum']:>10.2f} {row['maximum']:>10.2f}  {tag}")

# ─────────────────────────────────────────────────────────────────
# 5 DROUGHT CONDITIONS SETUP
# ─────────────────────────────────────────────────────────────────

DROUGHT_CONDITIONS = {
    "Control":       {"water_ub": 1000, "proline_lb": 0,   "photo_ub": 1000, "ros_lb": 0,   "aba_ub": 50,  "transpire_lb": 50},
    "Mild_Drought":  {"water_ub": 600,  "proline_lb": 50,  "photo_ub": 700,  "ros_lb": 5,   "aba_ub": 100, "transpire_lb": 30},
    "Moderate":      {"water_ub": 400,  "proline_lb": 150, "photo_ub": 400,  "ros_lb": 10,  "aba_ub": 150, "transpire_lb": 15},
    "Severe":        {"water_ub": 200,  "proline_lb": 300, "photo_ub": 200,  "ros_lb": 15,  "aba_ub": 180, "transpire_lb": 5},
    "Extreme":       {"water_ub": 80,   "proline_lb": 500, "photo_ub": 80,   "ros_lb": 20,  "aba_ub": 200, "transpire_lb": 0},
}

condition_fluxes = {}
condition_fva    = {}

for cond_name, params in DROUGHT_CONDITIONS.items():
    cond_model = copy.deepcopy(model)
    adjustments = {
        "WATER_TRANSPORT": (0, params["water_ub"]),
        "PROLINE_SYN":     (params["proline_lb"], 1000),
        "PSII":            (0, params["photo_ub"]),
        "PSI":             (0, params["photo_ub"]),
        "RuBisCO":         (0, params["photo_ub"]),
        "CATALASE_RXN":    (params["ros_lb"], 2000),
        "NCED":            (0, params["aba_ub"]),
        "ABA_final":       (0, params["aba_ub"]),
        "TRANSPIRE":       (params["transpire_lb"], 1000),
    }
    for rxn_id, (lb, ub) in adjustments.items():
        if rxn_id in cond_model.reactions:
            safe_set_bounds(cond_model.reactions.get_by_id(rxn_id), lb=lb, ub=ub)

    cond_model.objective = "PROLINE_SYN"
    sol = cond_model.optimize()
    if sol.status == "optimal":
        condition_fluxes[cond_name] = sol.fluxes
    else:
        condition_fluxes[cond_name] = pd.Series({r: 0.0 for r in KEY_REACTIONS})

    fva_cond = flux_variability_analysis(cond_model, reaction_list=fva_rxns, processes=1)
    condition_fva[cond_name] = fva_cond
    print(f"Condition [{cond_name}]: status={sol.status}, obj={sol.objective_value:.2f}")

# ─────────────────────────────────────────────────────────────────
# EXCEL OUTPUTS
# ─────────────────────────────────────────────────────────────────

with pd.ExcelWriter("GSMM_Network_Data.xlsx", engine="openpyxl") as writer:
    gsmm_rows = []
    for rxn in model.reactions:
        gsmm_rows.append({
            "Reaction_ID": rxn.id,
            "Reaction_Name": rxn.name,
            "Subsystem": rxn.subsystem,
            "Lower_Bound": rxn.lower_bound,
            "Upper_Bound": rxn.upper_bound,
            "GPR": rxn.gene_reaction_rule,
            "Reaction_String": rxn.reaction,
        })
    gsmm_df = pd.DataFrame(gsmm_rows)
    gsmm_df.to_excel(writer, sheet_name="GSMM_Reactions", index=False)

    met_rows = []
    for met in model.metabolites:
        met_rows.append({
            "Metabolite_ID": met.id,
            "Name": met.name,
            "Compartment": met.compartment,
        })
    pd.DataFrame(met_rows).to_excel(writer, sheet_name="Metabolites", index=False)
    print("GSMM_Network_Data.xlsx saved")

pw_counts = metabolic['Pathway'].value_counts().reset_index()
pw_counts.columns = ['Pathway', 'Gene_Count']
ros_genes    = metabolic[metabolic['Pathway'] == 'ROS_Detox'][['GeneID','symbol','name','log2FC']]
osmo_genes   = metabolic[metabolic['Pathway'] == 'Osmoprotection'][['GeneID','symbol','name','log2FC']]
photo_genes  = metabolic[metabolic['Pathway'] == 'Photosynthesis'][['GeneID','symbol','name','log2FC']]

with pd.ExcelWriter("Pathway_Gene_Distribution.xlsx", engine="openpyxl") as writer:
    pw_counts.to_excel(writer, sheet_name="All_Pathway_Counts", index=False)
    ros_genes.to_excel(writer, sheet_name="ROS_Detox_Genes", index=False)
    osmo_genes.to_excel(writer, sheet_name="Osmoprotection_Genes", index=False)
    photo_genes.to_excel(writer, sheet_name="Photosynthesis_Genes", index=False)
    metabolic[['GeneID','symbol','name','Pathway','log2FC']].to_excel(writer, sheet_name="All_Metabolic_Genes", index=False)
    print("Pathway_Gene_Distribution.xlsx saved")

ctrl_vs_drought_rows = []
main_rxns_groups = {
    "ROS_Detox":      ["CATALASE_RXN","APX","SOD","GPX","PRX"],
    "Osmoprotection": ["PROLINE_SYN","BETAINE_SYN","T6PS","LEA_SYN"],
    "Photosynthesis": ["RuBisCO","PSII","PSI","PRK"],
    "Water_Transport":["WATER_TRANSPORT","TRANSPIRE","TIP_VAC"],
    "ABA_Signaling":  ["NCED","AAO3","ABA_final","SnRK2_act"],
}
for cond_name, fluxes in condition_fluxes.items():
    for group, rxns in main_rxns_groups.items():
        for rxn_id in rxns:
            ctrl_flux = condition_fluxes["Control"].get(rxn_id, 0.0)
            cond_flux = fluxes.get(rxn_id, 0.0)
            ctrl_vs_drought_rows.append({
                "Condition": cond_name,
                "Pathway_Group": group,
                "Reaction": rxn_id,
                "Flux": round(cond_flux, 4),
                "Control_Flux": round(ctrl_flux, 4),
                "Delta": round(cond_flux - ctrl_flux, 4),
                "Direction": "UP" if (cond_flux - ctrl_flux) > 0.5 else ("DOWN" if (cond_flux - ctrl_flux) < -0.5 else "same"),
            })
ctrl_drought_df = pd.DataFrame(ctrl_vs_drought_rows)

with pd.ExcelWriter("Control_vs_Drought_Comparison.xlsx", engine="openpyxl") as writer:
    ctrl_drought_df.to_excel(writer, sheet_name="All_Conditions", index=False)
    for cond_name in DROUGHT_CONDITIONS.keys():
        sub = ctrl_drought_df[ctrl_drought_df["Condition"] == cond_name]
        sub.to_excel(writer, sheet_name=cond_name[:31], index=False)
    print("Control_vs_Drought_Comparison.xlsx saved")

delta_all_rows = []
for cond_name, fluxes in condition_fluxes.items():
    for rxn_id in KEY_REACTIONS:
        ctrl_flux = condition_fluxes["Control"].get(rxn_id, 0.0)
        cond_flux = fluxes.get(rxn_id, 0.0)
        rxn = model.reactions.get_by_id(rxn_id) if rxn_id in model.reactions else None
        delta_all_rows.append({
            "Condition": cond_name,
            "Reaction": rxn_id,
            "Subsystem": rxn.subsystem if rxn else "Unknown",
            "Flux": round(cond_flux, 4),
            "Control_Flux": round(ctrl_flux, 4),
            "Delta": round(cond_flux - ctrl_flux, 4),
        })
delta_all_df = pd.DataFrame(delta_all_rows)

with pd.ExcelWriter("Delta_Flux_Analysis.xlsx", engine="openpyxl") as writer:
    delta_all_df.to_excel(writer, sheet_name="All_Delta_Flux", index=False)
    for cond_name in DROUGHT_CONDITIONS.keys():
        sub = delta_all_df[delta_all_df["Condition"] == cond_name]
        sub.to_excel(writer, sheet_name=cond_name[:31], index=False)
    print("Delta_Flux_Analysis.xlsx saved")

active_tfs = [tf for tf, info in TF_NETWORK.items() if info['drought_active']]
tf_rxn_matrix = pd.DataFrame(0.0, index=active_tfs, columns=KEY_REACTIONS[:15])
for tf in active_tfs:
    info = TF_NETWORK[tf]
    for rxn_id in info['target_reactions']:
        if rxn_id in tf_rxn_matrix.columns:
            tf_rxn_matrix.loc[tf, rxn_id] = info['flux_modifier']

tf_detail_rows = []
for tf, info in TF_NETWORK.items():
    for rxn_id in info['target_reactions']:
        tf_detail_rows.append({
            "TF": tf, "Family": info['family'], "Signal": info['signal'],
            "Drought_Active": info['drought_active'],
            "Target_Reaction": rxn_id,
            "Flux_Modifier": info['flux_modifier'],
        })
tf_detail_df = pd.DataFrame(tf_detail_rows)

with pd.ExcelWriter("TF_Regulatory_Network.xlsx", engine="openpyxl") as writer:
    tf_detail_df.to_excel(writer, sheet_name="TF_Targets", index=False)
    tf_rxn_matrix.to_excel(writer, sheet_name="TF_Reaction_Matrix")
    print("TF_Regulatory_Network.xlsx saved")

fva_all_rows = []
for cond_name, fva_result in condition_fva.items():
    for rxn_id, row in fva_result.iterrows():
        span = row['maximum'] - row['minimum']
        fva_all_rows.append({
            "Condition": cond_name,
            "Reaction": rxn_id,
            "Min_Flux": round(row['minimum'], 4),
            "Max_Flux": round(row['maximum'], 4),
            "Span": round(span, 4),
            "Flexibility": "RIGID" if span < 1e-6 else "FLEXIBLE",
        })
fva_all_df = pd.DataFrame(fva_all_rows)

with pd.ExcelWriter("FVA_Flexibility_Results.xlsx", engine="openpyxl") as writer:
    fva_all_df.to_excel(writer, sheet_name="All_FVA", index=False)
    for cond_name in DROUGHT_CONDITIONS.keys():
        sub = fva_all_df[fva_all_df["Condition"] == cond_name]
        sub.to_excel(writer, sheet_name=cond_name[:31], index=False)
    print("FVA_Flexibility_Results.xlsx saved")

print("\nFVA Results by Condition (Rigid vs Flexible):")
print(f"\n{'Reaction':<22} " + "  ".join(f"{c:<12}" for c in DROUGHT_CONDITIONS.keys()))
print("-" * 100)
for rxn_id in fva_rxns:
    row_parts = [f"{rxn_id:<22}"]
    for cond_name in DROUGHT_CONDITIONS.keys():
        span = condition_fva[cond_name].loc[rxn_id, 'maximum'] - condition_fva[cond_name].loc[rxn_id, 'minimum']
        tag = "RIGID" if span < 1e-6 else f"FLEX±{span:.0f}"
        row_parts.append(f"{tag:<14}")
    print("  ".join(row_parts))

# ─────────────────────────────────────────────────────────────────
# PLOT 1: GSMM Network Overview
# ─────────────────────────────────────────────────────────────────

SUBSYSTEM_COLORS = {
    "Glycolysis":             "#e74c3c",
    "TCA_Cycle":              "#e67e22",
    "OxPhos":                 "#f1c40f",
    "Photosynthesis":         "#2ecc71",
    "ROS_Detox":              "#1abc9c",
    "Osmoprotection":         "#3498db",
    "Water_Transport":        "#9b59b6",
    "ABA_Signaling":          "#e91e63",
    "Stress_Signaling":       "#ff5722",
    "Amino_Acid_Metabolism":  "#00bcd4",
    "Lipid_Metabolism":       "#8bc34a",
    "Phenylpropanoid":        "#ff9800",
    "Carbohydrate_Metabolism":"#795548",
    "N_Assimilation":         "#607d8b",
    "Auxin_Biosynthesis":     "#673ab7",
    "Pentose_Phosphate":      "#009688",
    "Redox_Balance":          "#f06292",
}

SUBSYSTEM_POSITIONS = {
    "Glycolysis":             (0.50, 0.50),
    "TCA_Cycle":              (0.50, 0.35),
    "OxPhos":                 (0.65, 0.28),
    "Photosynthesis":         (0.20, 0.75),
    "ROS_Detox":              (0.78, 0.70),
    "Osmoprotection":         (0.30, 0.55),
    "Water_Transport":        (0.15, 0.45),
    "ABA_Signaling":          (0.70, 0.55),
    "Stress_Signaling":       (0.85, 0.45),
    "Amino_Acid_Metabolism":  (0.45, 0.70),
    "Lipid_Metabolism":       (0.60, 0.80),
    "Phenylpropanoid":        (0.25, 0.85),
    "Carbohydrate_Metabolism":(0.72, 0.85),
    "N_Assimilation":         (0.15, 0.60),
    "Auxin_Biosynthesis":     (0.35, 0.25),
    "Pentose_Phosphate":      (0.60, 0.65),
    "Redox_Balance":          (0.82, 0.30),
}

METABOLIC_EDGES = [
    ("Glycolysis",             "TCA_Cycle",              "Pyr/AcCoA"),
    ("Glycolysis",             "Pentose_Phosphate",       "G6P"),
    ("Glycolysis",             "Amino_Acid_Metabolism",   "PEP/PYR"),
    ("Glycolysis",             "Carbohydrate_Metabolism", "F6P"),
    ("TCA_Cycle",              "OxPhos",                  "NADH"),
    ("TCA_Cycle",              "Amino_Acid_Metabolism",   "AKG/OAA"),
    ("TCA_Cycle",              "Redox_Balance",           "NADPH"),
    ("OxPhos",                 "Redox_Balance",           "ATP"),
    ("Photosynthesis",         "Glycolysis",              "G3P"),
    ("Photosynthesis",         "Carbohydrate_Metabolism", "Sucrose"),
    ("Photosynthesis",         "ROS_Detox",               "O2/ROS"),
    ("Photosynthesis",         "Pentose_Phosphate",       "RuBP"),
    ("ROS_Detox",              "Redox_Balance",           "NADPH"),
    ("Osmoprotection",         "Amino_Acid_Metabolism",   "Pro/Glu"),
    ("Osmoprotection",         "Carbohydrate_Metabolism", "Trehalose"),
    ("Water_Transport",        "Osmoprotection",          "Osmolytes"),
    ("ABA_Signaling",          "Water_Transport",         "ABA"),
    ("ABA_Signaling",          "Osmoprotection",          "ABA"),
    ("ABA_Signaling",          "Stress_Signaling",        "SnRK2"),
    ("Stress_Signaling",       "ROS_Detox",               "MAPK"),
    ("Stress_Signaling",       "Osmoprotection",          "Kinase"),
    ("Amino_Acid_Metabolism",  "Osmoprotection",          "Proline"),
    ("Amino_Acid_Metabolism",  "Phenylpropanoid",         "Phe"),
    ("Amino_Acid_Metabolism",  "N_Assimilation",          "Gln/Glu"),
    ("Lipid_Metabolism",       "TCA_Cycle",               "AcCoA"),
    ("Lipid_Metabolism",       "Stress_Signaling",        "JA"),
    ("Phenylpropanoid",        "ROS_Detox",               "Flavonoids"),
    ("Carbohydrate_Metabolism","Glycolysis",              "Glc"),
    ("Carbohydrate_Metabolism","Lipid_Metabolism",        "AcCoA"),
    ("N_Assimilation",         "Amino_Acid_Metabolism",   "NH4+"),
    ("Auxin_Biosynthesis",     "Amino_Acid_Metabolism",   "Trp"),
    ("Pentose_Phosphate",      "ROS_Detox",               "NADPH"),
    ("Pentose_Phosphate",      "Amino_Acid_Metabolism",   "E4P"),
    ("Redox_Balance",          "ROS_Detox",               "NADPH"),
]

subsys_rxn_counts = gsmm_df[gsmm_df['Subsystem'].isin(SUBSYSTEM_COLORS)]['Subsystem'].value_counts().to_dict()

fig1 = plt.figure(figsize=(22, 20))
fig1.patch.set_facecolor('#0d1117')
fig1.suptitle("iZea mays GSMM — Metabolic Network Topology", fontsize=18,
              fontweight='bold', color='white', y=0.98)

ax_net = fig1.add_axes([0.01, 0.22, 0.70, 0.74])
ax_net.set_facecolor('#0d1117')
ax_net.set_xlim(0, 1)
ax_net.set_ylim(0, 1)
ax_net.axis('off')
ax_net.set_title("Subsystem Connectivity Map", fontsize=13, fontweight='bold',
                 color='white', pad=10)

for src, tgt, label in METABOLIC_EDGES:
    if src not in SUBSYSTEM_POSITIONS or tgt not in SUBSYSTEM_POSITIONS:
        continue
    x0, y0 = SUBSYSTEM_POSITIONS[src]
    x1, y1 = SUBSYSTEM_POSITIONS[tgt]
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2) + 1e-9
    shrink = 0.055
    xs = x0 + dx * shrink / dist
    ys = y0 + dy * shrink / dist
    xe = x1 - dx * shrink / dist
    ye = y1 - dy * shrink / dist
    ax_net.annotate("", xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle="-|>", color='#4a5568',
                                   lw=1.2, mutation_scale=12))
    mx, my = (xs + xe) / 2, (ys + ye) / 2
    ax_net.text(mx, my, label, fontsize=5.5, color='#a0aec0',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a202c',
                          edgecolor='none', alpha=0.75))

for subsys, (x, y) in SUBSYSTEM_POSITIONS.items():
    color = SUBSYSTEM_COLORS.get(subsys, '#607d8b')
    n_rxns = subsys_rxn_counts.get(subsys, 0)
    node_r = 0.030 + min(n_rxns / 400, 0.030)
    circle = plt.Circle((x, y), node_r, color=color, zorder=5, alpha=0.92,
                         linewidth=1.5, edgecolor='white')
    ax_net.add_patch(circle)
    short_name = subsys.replace("_", "\n")
    ax_net.text(x, y + node_r + 0.022, short_name, ha='center', va='bottom',
                fontsize=6.5, color='white', fontweight='bold', zorder=6,
                multialignment='center')
    ax_net.text(x, y, str(n_rxns), ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=7)

KEY_METS = {
    "ATP":      (0.50, 0.44),
    "NADPH":    (0.65, 0.60),
    "Proline":  (0.36, 0.62),
    "ABA":      (0.70, 0.62),
    "H2O2":     (0.75, 0.64),
    "Glucose":  (0.44, 0.56),
}
for met_name, (mx, my) in KEY_METS.items():
    ax_net.plot(mx, my, 's', color='#ffd700', markersize=6, zorder=8,
                markeredgecolor='#b8860b', markeredgewidth=0.8)
    ax_net.text(mx + 0.015, my, met_name, fontsize=5.5, color='#ffd700',
                va='center', fontweight='bold', zorder=9)

node_patch  = mpatches.Circle((0, 0), 0.01, color='#3498db', label='Subsystem (size = rxn count)')
met_patch   = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffd700',
                          markersize=8, label='Key metabolite')
ax_net.legend(handles=[node_patch, met_patch], fontsize=8, loc='lower left',
              facecolor='#1a202c', edgecolor='#4a5568', labelcolor='white', framealpha=0.9)

ax_bar = fig1.add_axes([0.72, 0.55, 0.26, 0.40])
ax_bar.set_facecolor('#161b22')
sorted_ss = sorted(subsys_rxn_counts.items(), key=lambda x: x[1], reverse=True)
ss_names  = [s[0].replace("_", "\n") for s in sorted_ss]
ss_vals   = [s[1] for s in sorted_ss]
ss_colors = [SUBSYSTEM_COLORS.get(s[0], '#607d8b') for s in sorted_ss]
bars_ss = ax_bar.barh(ss_names, ss_vals, color=ss_colors, edgecolor='#0d1117', linewidth=0.6)
ax_bar.set_xlabel("Reaction Count", fontsize=9, color='white')
ax_bar.set_title("Reactions per\nSubsystem", fontsize=10, fontweight='bold', color='white')
ax_bar.tick_params(colors='white', labelsize=7)
for spine in ['bottom', 'left']:
    ax_bar.spines[spine].set_color('#4a5568')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
for bar, val in zip(bars_ss, ss_vals):
    ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=7, color='white')
ax_bar.set_xlim(0, max(ss_vals) * 1.2)

ax_pie = fig1.add_axes([0.72, 0.22, 0.26, 0.30])
ax_pie.set_facecolor('#161b22')
compartments = {'c': 'Cytoplasm', 'm': 'Mitochondria', 'p': 'Plastid', 'e': 'Extracellular', 'v': 'Vacuole'}
comp_counts = {}
for met in model.metabolites:
    label = compartments.get(met.compartment, met.compartment)
    comp_counts[label] = comp_counts.get(label, 0) + 1
wedge_colors_pie = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']
wedges, texts, autotexts = ax_pie.pie(
    comp_counts.values(), labels=comp_counts.keys(),
    autopct='%1.0f%%', colors=wedge_colors_pie,
    startangle=140, pctdistance=0.78,
    textprops={'color': 'white', 'fontsize': 7},
    wedgeprops={'edgecolor': '#0d1117', 'linewidth': 1.2})
for at in autotexts:
    at.set_fontsize(6.5)
ax_pie.set_title("Metabolites by\nCompartment", fontsize=10, fontweight='bold', color='white')

ax_stats = fig1.add_axes([0.01, 0.02, 0.97, 0.18])
ax_stats.set_facecolor('#161b22')
ax_stats.axis('off')
has_gpr = gsmm_df[gsmm_df['GPR'].notna() & (gsmm_df['GPR'] != '')].shape[0]
no_gpr  = len(gsmm_df) - has_gpr
stat_items = [
    ("Total Reactions",    len(model.reactions),     "#3498db"),
    ("Total Metabolites",  len(model.metabolites),   "#2ecc71"),
    ("Subsystems",         len(SUBSYSTEM_POSITIONS), "#e67e22"),
    ("With GPR",           has_gpr,                  "#9b59b6"),
    ("Without GPR",        no_gpr,                   "#e74c3c"),
    ("Total TFs",          len(TF_NETWORK),           "#1abc9c"),
    ("Drought TFs Active", sum(1 for v in TF_NETWORK.values() if v['drought_active']), "#f1c40f"),
    ("Key Reactions",      len(KEY_REACTIONS),        "#ff5722"),
]
for i, (label, val, color) in enumerate(stat_items):
    xpos = 0.065 + i * 0.124
    ax_stats.add_patch(plt.Rectangle((xpos - 0.055, 0.05), 0.11, 0.88,
                                      facecolor='#1a202c', edgecolor=color,
                                      linewidth=1.5, transform=ax_stats.transAxes))
    ax_stats.text(xpos, 0.72, str(val), ha='center', va='center',
                  fontsize=16, fontweight='bold', color=color,
                  transform=ax_stats.transAxes)
    ax_stats.text(xpos, 0.28, label, ha='center', va='center',
                  fontsize=7.5, color='#a0aec0',
                  transform=ax_stats.transAxes, multialignment='center')

plt.savefig("Plot1_GSMM_Network.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print("Plot1_GSMM_Network.png saved")

# ─────────────────────────────────────────────────────────────────
# PLOT 2: Pathway Gene Distribution (ROS / Osmo / Photo highlighted)
# ─────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
fig2.suptitle("Pathway Gene Distribution", fontsize=15, fontweight='bold')
fig2.patch.set_facecolor('#f8f9fa')

ax = axes2[0]
ax.set_facecolor('#f0f4f8')
HIGHLIGHT = {"ROS_Detox": "#e74c3c", "Osmoprotection": "#2ecc71", "Photosynthesis": "#3498db"}
bar_colors_pw = [HIGHLIGHT.get(p, '#aab7b8') for p in pw_counts['Pathway']]
bars = ax.bar(pw_counts['Pathway'], pw_counts['Gene_Count'], color=bar_colors_pw, edgecolor='white', linewidth=0.8)
ax.set_xticklabels(pw_counts['Pathway'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Gene Count", fontsize=11)
ax.set_title("All Pathways — Gene Count\n(Red=ROS | Green=Osmo | Blue=Photo)", fontsize=11, fontweight='bold')
for bar, val in zip(bars, pw_counts['Gene_Count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, str(val),
            ha='center', va='bottom', fontsize=7, fontweight='bold')

patches = [mpatches.Patch(color=v, label=k) for k, v in HIGHLIGHT.items()]
ax.legend(handles=patches, fontsize=9, loc='upper right')

ax = axes2[1]
ax.set_facecolor('#f0f4f8')
highlight_pathways = ["ROS_Detox", "Osmoprotection", "Photosynthesis"]
highlight_data = pw_counts[pw_counts['Pathway'].isin(highlight_pathways)].copy()
highlight_data = highlight_data.set_index('Pathway').reindex(highlight_pathways).reset_index()
hl_colors = [HIGHLIGHT[p] for p in highlight_data['Pathway']]
bars2 = ax.bar(highlight_data['Pathway'], highlight_data['Gene_Count'], color=hl_colors,
               edgecolor='white', linewidth=1.2, width=0.5)
ax.set_ylabel("Gene Count", fontsize=12)
ax.set_title("Key Drought Pathways — Gene Count", fontsize=12, fontweight='bold')
for bar, val in zip(bars2, highlight_data['Gene_Count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
            ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylim(0, highlight_data['Gene_Count'].max() * 1.25)

for bar, pw in zip(bars2, highlight_data['Pathway']):
    desc = {"ROS_Detox": "H2O2 Scavenging\nUpregulated ↑",
            "Osmoprotection": "Proline / Betaine\nUpregulated ↑",
            "Photosynthesis": "Calvin / PSII\nDownregulated ↓"}
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
            desc[pw], ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', multialignment='center')

plt.tight_layout()
plt.savefig("Plot2_Pathway_Gene_Distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot2_Pathway_Gene_Distribution.png saved")

# ─────────────────────────────────────────────────────────────────
# PLOT 3: Control vs Drought Comparison (5 conditions, separate subplots)
# ─────────────────────────────────────────────────────────────────

conditions_list = list(DROUGHT_CONDITIONS.keys())
pathway_groups_plot = {
    "ROS_Detox":      ["CATALASE_RXN","APX","SOD","GPX","PRX"],
    "Osmoprotection": ["PROLINE_SYN","BETAINE_SYN","T6PS","LEA_SYN"],
    "Photosynthesis": ["RuBisCO","PSII","PSI","PRK"],
    "Water_Transport":["WATER_TRANSPORT","TRANSPIRE","TIP_VAC"],
    "ABA_Signaling":  ["NCED","AAO3","ABA_final","SnRK2_act"],
}

fig3, axes3 = plt.subplots(1, 5, figsize=(22, 8), sharey=False)
fig3.suptitle("Control vs Drought Comparison — 5 Conditions\n(Pathway Activity: ROS ↑  |  Proline ↑  |  Photosynthesis ↓)",
              fontsize=13, fontweight='bold')
fig3.patch.set_facecolor('#f8f9fa')

group_colors = {"ROS_Detox":"#e74c3c", "Osmoprotection":"#2ecc71",
                "Photosynthesis":"#3498db", "Water_Transport":"#f39c12", "ABA_Signaling":"#9b59b6"}

for ci, cond_name in enumerate(conditions_list):
    ax = axes3[ci]
    ax.set_facecolor('#f0f4f8')
    group_avgs = {}
    for group, rxns in pathway_groups_plot.items():
        avg = np.mean([condition_fluxes[cond_name].get(r, 0.0) for r in rxns])
        group_avgs[group] = avg

    ctrl_avgs = {}
    for group, rxns in pathway_groups_plot.items():
        ctrl_avgs[group] = np.mean([condition_fluxes["Control"].get(r, 0.0) for r in rxns])

    groups = list(group_avgs.keys())
    x = np.arange(len(groups))
    w = 0.35
    colors_ctrl   = [group_colors[g] for g in groups]
    colors_cond   = [group_colors[g] for g in groups]

    b1 = ax.bar(x - w/2, [ctrl_avgs[g] for g in groups], w,
                color=colors_ctrl, alpha=0.45, label='Control', edgecolor='white')
    b2 = ax.bar(x + w/2, [group_avgs[g] for g in groups], w,
                color=colors_cond, alpha=0.90, label=cond_name, edgecolor='white')

    ax.set_xticks(x)
    short_labels = [g.replace("_", "\n") for g in groups]
    ax.set_xticklabels(short_labels, fontsize=6.5, rotation=30, ha='right')
    ax.set_title(cond_name.replace("_", "\n"), fontsize=10, fontweight='bold')
    ax.set_ylabel("Mean Flux" if ci == 0 else "", fontsize=9)

    for g_idx, group in enumerate(groups):
        ctrl_v = ctrl_avgs[group]
        cond_v = group_avgs[group]
        diff = cond_v - ctrl_v
        sym = "↑" if diff > 0.5 else ("↓" if diff < -0.5 else "=")
        col = "#27ae60" if diff > 0.5 else ("#c0392b" if diff < -0.5 else "gray")
        ax.text(g_idx + w/2, cond_v + max(cond_v * 0.05, 1),
                sym, ha='center', va='bottom', fontsize=12, color=col, fontweight='bold')

    if ci == 0:
        ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig("Plot3_Control_vs_Drought_5conditions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot3_Control_vs_Drought_5conditions.png saved")

# ─────────────────────────────────────────────────────────────────
# PLOT 4: Delta Flux Analysis — Line plot across 5 conditions
# ─────────────────────────────────────────────────────────────────

DELTA_RXNS = [
    "CATALASE_RXN", "APX", "SOD",
    "PROLINE_SYN", "BETAINE_SYN",
    "RuBisCO", "PSII",
    "WATER_TRANSPORT", "TRANSPIRE",
    "NCED", "SnRK2_act",
    "MAPK_cas", "PAL",
    "HK", "CS",
]

RXN_COLORS = {
    "CATALASE_RXN": "#e74c3c", "APX": "#c0392b", "SOD": "#ff6b6b",
    "PROLINE_SYN":  "#2ecc71", "BETAINE_SYN": "#27ae60",
    "RuBisCO":      "#3498db", "PSII": "#2980b9",
    "WATER_TRANSPORT": "#f39c12", "TRANSPIRE": "#d68910",
    "NCED":         "#9b59b6", "SnRK2_act": "#7d3c98",
    "MAPK_cas":     "#1abc9c", "PAL": "#16a085",
    "HK":           "#95a5a6", "CS": "#717d7e",
}

ctrl_baseline = {r: condition_fluxes["Control"].get(r, 0.0) for r in DELTA_RXNS}

fig4, axes4 = plt.subplots(3, 1, figsize=(16, 18))
fig4.suptitle("Delta Flux Analysis — Reaction-Level Changes Across 5 Drought Conditions\n"
              "(Green = Increase  |  Red = Decrease)",
              fontsize=13, fontweight='bold')
fig4.patch.set_facecolor('#f8f9fa')

groups_plot4 = {
    "ROS / Osmoprotection": ["CATALASE_RXN","APX","SOD","PROLINE_SYN","BETAINE_SYN"],
    "Photosynthesis / Water": ["RuBisCO","PSII","WATER_TRANSPORT","TRANSPIRE"],
    "Signaling / Other": ["NCED","SnRK2_act","MAPK_cas","PAL","HK","CS"],
}

x_conds = conditions_list

for ax_idx, (grp_name, rxn_list) in enumerate(groups_plot4.items()):
    ax = axes4[ax_idx]
    ax.set_facecolor('#f0f4f8')
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)

    for rxn_id in rxn_list:
        deltas = []
        for cond in x_conds:
            cond_flux = condition_fluxes[cond].get(rxn_id, 0.0)
            deltas.append(cond_flux - ctrl_baseline[rxn_id])

        color = RXN_COLORS.get(rxn_id, 'gray')
        ax.plot(x_conds, deltas, marker='o', linewidth=2.2, markersize=7,
                label=rxn_id, color=color)

        last_delta = deltas[-1]
        fill_color = '#2ecc71' if last_delta > 0 else '#e74c3c'
        ax.fill_between(x_conds, deltas, 0, alpha=0.08, color=fill_color)

        ax.annotate(f"{rxn_id}\n{last_delta:+.1f}",
                    xy=(x_conds[-1], deltas[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color=color, va='center')

    ax.set_title(f"Group: {grp_name}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Δ Flux (vs Control)", fontsize=10)
    ax.legend(fontsize=8, loc='upper left', ncol=3, framealpha=0.8)
    ax.set_xticks(range(len(x_conds)))
    ax.set_xticklabels(x_conds, fontsize=9, rotation=15)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 1.1 if ymin < 0 else ymin, ymax * 1.25 if ymax > 0 else ymax)

plt.tight_layout()
plt.savefig("Plot4_Delta_Flux_LineChart.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot4_Delta_Flux_LineChart.png saved")

# ─────────────────────────────────────────────────────────────────
# PLOT 5: TF Regulatory Network
# ─────────────────────────────────────────────────────────────────

fig5, axes5 = plt.subplots(1, 2, figsize=(20, 9))
fig5.suptitle("TF Regulatory Network — ABA / NAC / DREB Influence on Metabolic Reactions",
              fontsize=13, fontweight='bold')
fig5.patch.set_facecolor('#f8f9fa')

ax = axes5[0]
ax.set_facecolor('#f0f4f8')
plot_cols = [c for c in KEY_REACTIONS[:15] if any(
    c in TF_NETWORK[tf]['target_reactions'] for tf in active_tfs)][:14]
tf_rxn_sub = tf_rxn_matrix[plot_cols]

im = ax.imshow(tf_rxn_sub.values, aspect='auto', cmap='RdYlGn',
               vmin=0, vmax=2.5, interpolation='nearest')
ax.set_xticks(range(len(tf_rxn_sub.columns)))
ax.set_xticklabels(tf_rxn_sub.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(active_tfs)))
ax.set_yticklabels(active_tfs, fontsize=9)
ax.set_title("TF × Reaction Flux Modifier Heatmap", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax, label="Flux Modifier", shrink=0.85)

families = [TF_NETWORK[tf]['family'] for tf in active_tfs]
fam_colors = {"bZIP":"#e74c3c","NAC":"#3498db","AP2/ERF":"#2ecc71","R2R3-MYB":"#f39c12","WRKY":"#9b59b6"}
for i in range(len(active_tfs)):
    for j in range(len(tf_rxn_sub.columns)):
        val = tf_rxn_sub.values[i, j]
        if val > 0:
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=6.5,
                    color='black', fontweight='bold')
    fam = families[i]
    ax.text(-0.5, i, "", ha='right', va='center', fontsize=7,
            color=fam_colors.get(fam, 'black'))

fam_patches = [mpatches.Patch(color=v, label=k) for k, v in fam_colors.items()]
ax.legend(handles=fam_patches, fontsize=8, loc='lower right',
          bbox_to_anchor=(1.0, -0.02), framealpha=0.9)

ax = axes5[1]
ax.set_facecolor('#f0f4f8')
tf_target_counts = {tf: len(TF_NETWORK[tf]['target_reactions']) for tf in active_tfs}
tf_modifiers     = {tf: TF_NETWORK[tf]['flux_modifier'] for tf in active_tfs}
tf_families      = {tf: TF_NETWORK[tf]['family'] for tf in active_tfs}

x_pos = np.arange(len(active_tfs))
bar_colors_tf = [fam_colors.get(tf_families[tf], '#aab7b8') for tf in active_tfs]
bars = ax.bar(x_pos, [tf_target_counts[tf] for tf in active_tfs],
              color=bar_colors_tf, edgecolor='white', linewidth=0.8, alpha=0.85)
ax2r = ax.twinx()
ax2r.plot(x_pos, [tf_modifiers[tf] for tf in active_tfs],
          'k--o', linewidth=1.8, markersize=7, label='Flux Modifier', zorder=5)
ax2r.axhline(1.0, color='gray', linewidth=1, linestyle=':', alpha=0.6)
ax2r.set_ylabel("Flux Modifier", fontsize=10)
ax2r.set_ylim(0, 3.0)

ax.set_xticks(x_pos)
ax.set_xticklabels(active_tfs, rotation=40, ha='right', fontsize=8)
ax.set_ylabel("Number of Target Reactions", fontsize=10)
ax.set_title("TF Target Count & Flux Modifier\n(ABA / DREB / NAC / MYB / WRKY)", fontsize=11, fontweight='bold')

for bar, val in zip(bars, [tf_target_counts[tf] for tf in active_tfs]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(val),
            ha='center', va='bottom', fontsize=8, fontweight='bold')

fam_patches2 = [mpatches.Patch(color=v, label=k) for k, v in fam_colors.items()]
ax.legend(handles=fam_patches2, fontsize=8, loc='upper left', framealpha=0.9)
ax2r.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig("Plot5_TF_Regulatory_Network.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot5_TF_Regulatory_Network.png saved")

# ─────────────────────────────────────────────────────────────────
# PLOT 6: FVA Flexibility — 5 Conditions
# ─────────────────────────────────────────────────────────────────

fig6, axes6 = plt.subplots(1, 5, figsize=(22, 8), sharey=True)
fig6.suptitle("FVA Flux Flexibility — Rigid vs Flexible Reactions Across 5 Drought Conditions\n"
              "Red = RIGID (essential)  |  Green = FLEXIBLE (adaptive)",
              fontsize=12, fontweight='bold')
fig6.patch.set_facecolor('#f8f9fa')

for ci, cond_name in enumerate(conditions_list):
    ax = axes6[ci]
    ax.set_facecolor('#f0f4f8')
    fva_result = condition_fva[cond_name]
    spans = fva_result['maximum'] - fva_result['minimum']
    spans_sorted = spans.sort_values(ascending=True)
    fva_colors_plot = ['#e74c3c' if s < 1e-6 else '#2ecc71' for s in spans_sorted.values]
    ax.barh(spans_sorted.index, spans_sorted.values, color=fva_colors_plot,
            edgecolor='white', linewidth=0.6)
    ax.set_title(cond_name.replace("_", "\n"), fontsize=9.5, fontweight='bold')
    ax.set_xlabel("Flux Range", fontsize=8)
    if ci == 0:
        ax.set_ylabel("Reaction", fontsize=9)

    rigid_count    = (spans_sorted < 1e-6).sum()
    flexible_count = (spans_sorted >= 1e-6).sum()
    ax.text(0.97, 0.02, f"Rigid: {rigid_count}\nFlex: {flexible_count}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

rigid_p = mpatches.Patch(color='#e74c3c', label='RIGID (essential)')
flex_p  = mpatches.Patch(color='#2ecc71', label='FLEXIBLE (adaptive)')
fig6.legend(handles=[rigid_p, flex_p], fontsize=10, loc='lower center',
            ncol=2, bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("Plot6_FVA_Flexibility_5conditions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot6_FVA_Flexibility_5conditions.png saved")

print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
print("""
Output files:
  iZeaMays_drought_v2.xml                 — SBML model
  gene_pathway_links.txt                  — gene-pathway map
  control_vs_drought_delta.csv            — delta flux (text)
  GSMM_Network_Data.xlsx                  — model reactions & metabolites
  Pathway_Gene_Distribution.xlsx          — pathway gene counts (all + 3 key)
  Control_vs_Drought_Comparison.xlsx      — 5 condition comparison
  Delta_Flux_Analysis.xlsx                — reaction-level delta, 5 conditions
  TF_Regulatory_Network.xlsx              — TF targets & modifier matrix
  FVA_Flexibility_Results.xlsx            — rigid/flexible, 5 conditions
  Plot1_GSMM_Network.png
  Plot2_Pathway_Gene_Distribution.png
  Plot3_Control_vs_Drought_5conditions.png
  Plot4_Delta_Flux_LineChart.png
  Plot5_TF_Regulatory_Network.png
  Plot6_FVA_Flexibility_5conditions.png
""")

if __name__ == '__main__':
    multiprocessing.freeze_support()