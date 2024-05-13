---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# NumPy vs APL
![](https://assets-global.website-files.com/64483ba2cd3664f5275be1da/6453af3005d3267ad3bccce9_red-logo-short.svg)
![](https://assets-global.website-files.com/64483ba2cd3664f5275be1da/6453a4df70dfe3083a2fdc5d_logo-long.svg)
@MaxCan-Code "Max Sun APL"

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

# NumPy vs APL
- Going slow and easy, we have an hour
- APL NumPy side-by-side
- Goal: aware of APL as another tool
- Non-goal: switch
- Cover briefly then come back for Qs
- Intentionally left out details so please ask as I go

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

## About me
- Applied math in school
- Knew APL from YouTube
- Learned APL on the job last Jan (https://bcaresearch.com customer of Dyalog)

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

## What we'll do
1. Simple Moore's Law model
1. Real data (CSV)
1. Perform linear regression and predict exponential growth using ordinary least squares (tldr line of best fit)
- Compare exponential growth constants between models
- Save and share:
    - NumPy zipped file `*.npz`
    - APL namespace file `*.apln`
    - CSV
- Chat, happy little demos?

Modified from https://numpy.org/numpy-tutorials/content/mooreslaw-tutorial.html

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

## What we need
- Python (packages):
    - NumPy
    - Matplotlib
- Dyalog APL:
    - SharpPlot (included) (OOP)
    - Py'n'APL (optional, external, open source)

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

APL trigger warning
- Butchered NumPy-style APL for comparison
- APL users, sorry

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
import matplotlib.pyplot as plt
import numpy as np
from pynapl import APL
apl = APL.APL()
run = apl.eval
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Model
`Moores_law(year) = (e^B_M) × (e^(A_M × year))`

`A_M = (log 2) ÷ 2`

`B_M = (log 2250) - A_M × 1971`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
     #        exp←*
     #        log←*○
     A_M = np.log(2) / 2
run("A_M ←     (⍟ 2) ÷ 2")

     B_M = np.log(2250) - A_M * 1971
run("B_M ←     (⍟ 2250) - A_M × 1971")

     Moores_law = lambda year: np.exp(B_M) * np.exp(A_M * year)
run("Moores_law ← {      year←⍵ ⋄  (* B_M) ×      *(A_M × year)} ⋄")

[A_M, B_M], run("A_M B_M")
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

In 1971, there were 2250 transistors. Use `Moores_law` to predict how many in 1973.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
ML_1971 = Moores_law(1971)
ML_1973 = Moores_law(1973)
print("In 1973, G. Moore expects {:.0f} transistors on Intels chips".format(ML_1973))
print("This is x{:.2f} more transistors than 1971".format(ML_1973 / ML_1971))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
ML_1971←Moores_law 1971
ML_1973←Moores_law 1973

_←'.format←⍕'
⎕←'In 1973, G. Moore expects ',(1↓0⍕ML_1973),' transistors on Intels chips'
⎕←'This is x',(1↓2⍕ML_1973÷ML_1971),' more transistors than 1971'
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Loading historical data
- Transistor count each year in CSV
- Inspect before load
- Save year & count columns to `data`.

Print first 10 rows of `transistor_data.csv`. The columns are

|Processor|MOS transistor count|Date of Introduction|Designer|MOSprocess|Area|
|---|---|---|---|---|---|
|Intel 4004 (4-bit  16-pin)|2250|1971|Intel|"10,000 nm"|12 mm²|
|...|...|...|...|...|...|

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
! head transistor_data.csv
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
run("""
⎕←↑'UTF-8'∘⎕UCS¨⎕UCS¨⎕SH'head transistor_data.csv'
⎕←''
⎕←↑⎕SH'head transistor_data.csv'
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Only need columns 2 & 3

`np.loadtxt`:
- `delimiter = ','`
- `usecols = [1,2]`: import cols 2 & 3
- `skiprows = 1`: skip header row

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
data = np.loadtxt("transistor_data.csv", delimiter=",", usecols=[1, 2], skiprows=1)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
csv←(⎕CSV⍠'Separator' ',')'transistor_data.csv'⍬ 4
data←1↓csv[;2 3]
""");

# csv = np.loadtxt("transistor_data.csv") # Value Error: could not convert string to float
# data = csv[1:, 1:3]
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

- Assign `data` to `year` and `transistor_count`
- Print first 10 values (`[:10]`)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
year = data[:, 1]  # grab the second column and assign
transistor_count = data[:, 0]  # grab the first column and assign

print("year:\t\t", year[:10])
print("trans. cnt:\t", transistor_count[:10])
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
year←data[;2]⊣'  ⍝ grab the second column and assign'
transistor_count←data[;1]⊣'  ⍝ grab the first column and assign'

⎕←'year:		',10↑year
⎕←'trans. cnt:	',10↑transistor_count ⋄
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Goal: solve `yi = A × year + B` for `A`, `B`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
yi = np.log(transistor_count)
run("yi←  ⍟ transistor_count");
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Use least squares

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
model = np.polynomial.Polynomial.fit(year, yi, deg=1)
model = model.convert()
model
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
_←'https://aplcart.info?q=linear%20fit'
_←'matrix div ⌹ ⎕÷'
model←yi⌹(1,⍪year)
(B A)←model
⎕←'x' ' ' '↦'(1↓9⍕B)'+'(1↓9⍕A)'x'
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

The individual parameters `A` and `B` are the coefficients of our linear model:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
B, A = model
# (B A)←model
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Increase rate = `e^(2 × A)`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"Rate of semiconductors added on a chip every 2 years: {np.exp(2 * A):.2f}")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
⎕←'Rate of semiconductors added on a chip every 2 years: ',1↓2⍕*2×A
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

Plot & compare:
- Least squares
- Moore's law
- Real world data

+++ {"editable": true, "slideshow": {"slide_type": ""}}

style sheet: `fivethirtyeight`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
transistor_count_predicted = np.exp(B) * np.exp(A * year)
transistor_Moores_law = Moores_law(year)
plt.style.use("fivethirtyeight")
plt.semilogy(year, transistor_count, "s", label="MOS transistor count")
plt.semilogy(year, transistor_count_predicted, label="linear regression")


plot1 = plt.plot(year, transistor_Moores_law, label="Moore's Law")
plt.title(
    "MOS transistor count per microprocessor\n"
    + "every two years \n"
    + "Transistor count was x{:.2f} higher".format(np.exp(A * 2))
)
plt.xlabel("year introduced")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.ylabel("# of transistors\nper microprocessor")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
plot1[0].figure
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
run("""
_←']Chart ⍝ useful, windows only'
_538←{_sp←⍵
    _←_sp.SetLineStyles do Causeway.LineStyle.Solid
    FromHtml←Causeway.ColorTranslator.{FromHtml'#',⍵}
    _←'FromHtml←{256⊥255,16⊥⍉3 2⍴¯1+⍵⍳⍨⎕D,⎕C ⎕A}'
    _←'https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/mpl-data/stylelib/fivethirtyeight.mplstyle'
    _←_sp.SetColors do⊂Causeway.{Color.(⍵)}FromHtml¨'008fd5' 'fc4f30' 'e5ae38' '6d904f' '8b8b8b' '810f7c'
    _←_sp.SetBackground do Causeway.{Color.(⍵)}FromHtml'f0f0f0'
    _←_sp.SetKeyBackground do Causeway.{Color.(⍵)}FromHtml'f0f0f0'
    _←_sp.SetAxisStyle do(Causeway.{Color.(⍵)}FromHtml'cbcbcb')Causeway.LineStyle.Solid 0.5
    _←_sp.SetGridLineStyle do(Causeway.{Color.(⍵)}FromHtml'cbcbcb')Causeway.LineStyle.Solid 0.5
    _←_sp.SetPenWidths do 3
    font←'DejaVu Sans'
    _←_sp.SetHeadingFont do font
    _←_sp.SetCaptionFont do font
    _←_sp.SetLabelFont do font
    _←_sp.SetKeyFont do font
    _sp.KeyStyle←Causeway.KeyStyles.(Boxed+Vertical+MiddleAlign)
    _sp.XAxisStyle←Causeway.XAxisStyles.CenteredCaption
    _sp.YAxisStyle←Causeway.YAxisStyles.CenteredCaption
    _sp.YLabelFormat←'0.0E00'
    _sp.ScatterPlotStyle←Causeway.ScatterPlotStyles.GridLines
    _sp.LineGraphStyle←Causeway.LineGraphStyles.GridLines
    _sp}
do←{⍎'⍺⍺ ⍵ ⋄ 0' ⋄ ⍺⍺}
'InitCauseway'⎕CY'sharpplot'
InitCauseway ⍬
sp←⎕NEW Causeway.SharpPlot
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
import IPython
IPython.display.SVG(run("""
transistor_count_predicted←(*B)×(*A×year)
transistor_Moores_law←Moores_law year
sp←_538 sp
_←sp.SetKeyText do'MOS transistor count' 'linear regression' 'Moore''s Law'
sp.Heading←'MOS transistor count per microprocessor',(⎕UCS 10),'every two years',(⎕UCS 10),'Transistor count was x ',(1↓2⍕*A×2),' higher'
sp.XCaption←'year introduced'
sp.YCaption←'# of transistors',(⎕UCS 10),'per microprocessor'
sp.YAxisStyle←sp.YAxisStyle+Causeway.YAxisStyles.LogScale
sp.KeyStyle←sp.KeyStyle+Causeway.KeyStyles.RightAlign
_←sp.SetMarkers do Causeway.Marker.Block
_←sp.SetMargins do 80 30 50 150
_←sp.SetXTickMarks do 10
_←sp.SetYTickMarks do 2
_←sp.DrawScatterPlot do(transistor_count)year
_←sp.DrawLineGraph do transistor_count_predicted year
_←sp.DrawLineGraph do transistor_Moores_law year
sp.RenderSvg ⍬
"""))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
IPython.display.SVG(run("sp.RenderSvg ⍬"))
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Zoom in on `year == 2017`, compare:
- Moore's law
- Average transistor count
- Real world data

Use `alpha=0.2`, opaque points mean overlap

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
transistor_count2017 = transistor_count[year == 2017]
print(
    transistor_count2017.max(), transistor_count2017.min(), transistor_count2017.mean()
)
y = np.linspace(2016.5, 2017.5)
your_model2017 = np.exp(B) * np.exp(A * y)
Moore_Model2017 = Moores_law(y)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
transistor_count2017←transistor_count[⍸year=2017]
_←'transistor_count2017←(year=2017)/transistor_count'
mean←{v←⍵ ⋄ (+/v)÷(≢v)}
⎕←(⌈/transistor_count2017)(⌊/transistor_count2017)(mean transistor_count2017)

y←2016.5 2017.5
your_model2017←(*B)×(*A×y)
Moore_Model2017←Moores_law y
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
plt.plot(
    2017 * np.ones(np.sum(year == 2017)),
    transistor_count2017,
    "ro",
    label="2017",
    alpha=0.2,
)
plt.plot(2017, transistor_count2017.mean(), "g+", markersize=20, mew=6)

plt.plot(y, your_model2017, label="Your prediction")
plot2 = plt.plot(y, Moore_Model2017, label="Moores law")
plt.ylabel("# of transistors\nper microprocessor")
plt.legend()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
plot2[0].figure
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
IPython.display.SVG(run("""
_←sp.Reset do ⍬
sp←_538 sp
_←sp.SetXRange do y
_←sp.SetYRange do⍎'2.4e10'
_←sp.SetXTickMarks do 0.2
_←sp.SetMarkers do⊂Causeway.Marker.(Dot Plus)
_←sp.SetMarkerColors do Causeway.Color.LightCoral
_←sp.SetMarkerScales do 6
_←sp.SetMargins do 5 15 55 20
_←sp.SetYTickMarks do⍎'0.5e10'
sp.YCaption←'# of transistors',(⎕UCS 10),'per microprocessor'
_←sp.DrawScatterPlot do(transistor_count2017)(2017×(+/year=2017)⍴1)
_←sp.SetMarkerColors do Causeway.Color.Green
_←sp.SetPenWidths do 4
_←sp.SetMarkerScales do 3
_←sp.SetKeyText do'2017' ' ' 'Your prediction' 'Moore''s Law'
_←sp.DrawScatterPlot do,¨(mean transistor_count2017)2017
sp←_538 sp
sp.KeyStyle←sp.KeyStyle+Causeway.KeyStyles.BottomAlign
_←sp.DrawLineGraph do your_model2017 y
_←sp.DrawLineGraph do Moore_Model2017 y
sp.RenderSvg ⍬
"""))
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
IPython.display.SVG(run("sp.RenderSvg ⍬"))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

- Least squares: close to mean
- Moore's law: close to max

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

- Least squares: close to mean
- Moore's law: close to max

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Share as zipped arrays and CSV
- `np.savez`: NumPy arrays for other Python sessions
- APL namespace (also possible: component file)
- `np.savetxt`, `⎕CSV`: CSV

`savez` with `notes=notes`

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
notes = "the arrays in this file are the result of a linear regression model\n"
notes += "the arrays include\nyear: year of manufacture\n"
notes += "transistor_count: number of transistors reported by manufacturers in a given year\n"
notes += "transistor_count_predicted: linear regression model = exp({:.2f})*exp({:.2f}*year)\n".format(
    B, A
)
notes += "transistor_Moores_law: Moores law =exp({:.2f})*exp({:.2f}*year)\n".format(
    B_M, A_M
)
notes += "regression_csts: linear regression constants A and B for log(transistor_count)=A*year+B"
print(notes)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
run("""
nl←⎕UCS 10
notes←'the arrays in this file are the result of a linear regression model',nl
notes,←'the arrays include',nl,'year: year of manufacture',nl
notes,←'transistor_count: number of transistors reported by manufacturers in a given year',nl
notes,←'transistor_count_predicted: linear regression model = exp(',(1↓2⍕B),')*exp(',(1↓2⍕A),'*year)',nl

notes,←'transistor_Moores_law: Moores law =exp(',(1↓2⍕B_M),')*exp(',(1↓2⍕A_M),'*year)',nl

notes,←'regression_csts: linear regression constants A and B for log(transistor_count)=A*year+B'
⎕←notes
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
np.savez(
    "mooreslaw_regression.npz",
    notes=notes,
    year=year,
    transistor_count=transistor_count,
    transistor_count_predicted=transistor_count_predicted,
    transistor_Moores_law=transistor_Moores_law,
    regression_csts=(A, B),
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
_ns←⎕NS'notes' 'year' 'transistor_count' 'transistor_count_predicted' 'transistor_Moores_law'
_ns.regression_csts←A B
_←'https://dyalog.github.io/link/4.0/API/Link.Fix'
_←(⊂⎕SE.Dyalog.Array.Serialise _ns)⎕NPUT'mooreslaw_regression.apln' 1

_←'"" "mooreslaw_regression"⎕SE.Link.Fix ⎕SE.Dyalog.Array.Serialise _ns'
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
results = np.load("mooreslaw_regression.npz")
run("""
results←⎕SE.Dyalog.Array.Deserialise⊃⎕NGET'mooreslaw_regression.apln'
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(      results["regression_csts"][1])
run("⎕←1↓13⍕results. regression_csts  [2]");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
! ls
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
run("⎕←↑⎕SH'ls'");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### CSV
- `np.savetxt` with `header=head`
- need 2D array

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
head = "the columns in this file are the result of a linear regression model\n"
head += "the columns include\nyear: year of manufacture\n"
head += "transistor_count: number of transistors reported by manufacturers in a given year\n"
head += "transistor_count_predicted: linear regression model = exp({:.2f})*exp({:.2f}*year)\n".format(
    B, A
)
head += "transistor_Moores_law: Moores law =exp({:.2f})*exp({:.2f}*year)\n".format(
    B_M, A_M
)
head += "year:, transistor_count:, transistor_count_predicted:, transistor_Moores_law:"
print(head)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
run("""
head←'the columns in this file are the result of a linear regression model',nl
head,←'the columns include',nl,'year: year of manufacture',nl
head,←'transistor_count: number of transistors reported by manufacturers in a given year',nl
head,←'transistor_count_predicted: linear regression model = exp(',(1↓2⍕B),')*exp(',(1↓2⍕A),'*year)',nl

head,←'transistor_Moores_law: Moores law =exp(',(1↓2⍕B_M),')*exp(',(1↓2⍕A_M),'*year)',nl

head,←'year:, transistor_count:, transistor_count_predicted:, transistor_Moores_law:'
⎕←head
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

Build 2D array by `np.block`ing 1D vectors with `np.newaxis` (1-col arrays) together

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(  year.shape)
run("⎕←⍴year");

print(   year[:,np.newaxis].shape)
run("⎕←⍴⍪year");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
output = np.block(
    [
        year[:, np.newaxis],
        transistor_count[:, np.newaxis],
        transistor_count_predicted[:, np.newaxis],
        transistor_Moores_law[:, np.newaxis],
    ]
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: fragment
---
run("""
output←year,transistor_count,transistor_count_predicted,⍪transistor_Moores_law
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
np.savetxt("mooreslaw_regression.csv", X=output, delimiter=",", header=head)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
! head mooreslaw_regression.csv
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: subslide
---
run("""
content←⊂(('^'⎕R'# &')head),nl,output(⎕CSV⍠'Separator' ',')''
_←content ⎕NPUT'mooreslaw_regression.csv' 1
""");
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
run("""
⎕←↑'UTF-8'∘⎕UCS¨⎕UCS¨⎕SH'head mooreslaw_regression.csv'
⎕←''
⎕←↑⎕SH'head mooreslaw_regression.csv'
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

## Functions we used

<style>
td, th {
   border: none!important;
   background-color: #ffffff;
}
</style>

|                    |    |                                      |
| ---                | ---| ---                                  |
| `np.loadtxt`       | ←→ | `⎕CSV` `[;2 3]` `1↓`                 |
| `np.log`           | ←→ | `⍟`                                  |
| `np.exp`           | ←→ | `*`                                  |
| `lambda`           | ←→ | `{⍵}`                                |
| `plt.semilogy`     | ←→ | `Causeway.YAxisStyles.LogScale`      |
| `plt.plot`         | ←→ | `sp.RenderSvg`, `sp.Draw*` (OOP)     |
| `x[:10]`           | ←→ | `10↑x`                               |
| `x[year == 2017]`  | ←→ | `(year=2017)/x` or `x[⍸year=2017]`   |
| `np.block`         | ←→ | `,` `⍪`                              |
| `np.newaxis`       | ←→ | `⍪`                                  |
| `np.savez`         | ←→ | `⎕SE.Dyalog.Array.Serialise` `⎕NPUT` |

## Some points
- Specific functions ←→ small composable functions (FP)
- Composing symbols: `⍟ ○*` `⌹ ⎕÷`
- Nested function calls ←→ shallow definitions (debugging context)
- Mature ecosystem (plot styles, exports) ←→ workarounds/hacks/rolling your own

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

## Some points
- Specific functions ←→ small composable functions (FP)
- Composing symbols: `⍟ ○*` `⌹ ⎕÷`
- Nested function calls ←→ shallow definitions (debugging context)
- Mature ecosystem (plot styles, exports) ←→ workarounds/hacks/rolling your own

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

# Chat, Q&A
happy little demos?

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

- Point free appendix
- 1-liner glass cleaner fractal
- Open emacs from RIDE, prefix completion, d.apln jump
- XML: 1-liner ]Defs workflow, https://apl.quest vids

@MaxCan-Code "Max Sun APL"

+++ {"editable": true, "slideshow": {"slide_type": "slide"}}

# Point-free (tacit) demo
- Opinionated takes
- Not liable for mental dmg

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

- `n+2` letter name/notation/definition/formula/src code?
- see as 1 unit (`poly morph ism` ←→  `polymorphism`):
  - `(B_M×⍨∘*⍨∘*A_M×⊢)`
- vs autocomplete

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
Moores_law = lambda year: np.exp(B_M) * np.exp(A_M * year)
Moores_law(1971), Moores_law(1973)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
run("""
_←'Moores_law = lambda year: np.exp(B_M) * np.exp(A_M * year)'
   Moores_law ← {      year←⍵ ⋄  (* B_M) ×      *(A_M × year)}
⎕←                                 (B_M  ×⍨∘*⍨∘ * A_M × ⊢)1971 1973
_←'                                └────────┘'

⎕←(B_M×⍨∘*⍨∘*A_M×⊢)1971 1973
⎕←Moores_law       1971 1973

_←'https://aplcart.info?q=split%20compose (looks better next ver)'
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

- `n+2` letter name/notation/definition/formula/src code?
- see as 1 unit (`poly morph ism` ←→  `polymorphism`):
  - `(⊣⌹1,∘⍪⊢)`
- vs autocomplete

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
model = np.polynomial.Polynomial.fit(year, yi, deg=1)
model = model.convert()
B, A = model
print(B ,A)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
run("""
⎕←yi⌹(1,⍪year)

⎕←yi  ⌹1, ⍪  year
⎕←yi(⊣⌹1,∘⍪⊢)year
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

- `n+2` letter name/notation/definition/formula/src code?
- see as 1 unit (`poly morph ism` ←→  `polymorphism`):
  - `(⌈/ , ⌊/ , +/÷≢)`
- vs autocomplete

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(
    transistor_count2017.max(), transistor_count2017.min(), transistor_count2017.mean()
)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
run("""
mean←{v←⍵ ⋄ (+/v)÷(≢v)}
⎕←(⌈/transistor_count2017)(⌊/transistor_count2017)(mean transistor_count2017)
⎕←(⌈/       ,              ⌊/            ,         +/÷≢) transistor_count2017

⎕←(⌈/,⌊/,+/÷≢)transistor_count2017
""");
```

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

## Related Work

- Alan J. Perlis. 1977. In praise of APL: a language for lyrical programming. SIGAPL APL Quote Quad 8, 2 (December 1977), 44–47. https://doi.org/10.1145/586015.586019
- `J`: primary parts of speech:

|        |    |                   |
| ---    | ---| ---               |
| noun   | ←→ | data              |
| verb   | ←→ | function on nouns |
| adverb | ←→ | modify verb       |
https://code.jsoftware.com/wiki/Vocabulary/Words#Parts_Of_Speech
