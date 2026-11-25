# The Long Way Up

Qwen3.8-Flash-Next drew and animated a two-burn transfer between Earth orbits from a single prompt. Running locally on an RTX Spark laptop with FP4-Local weights, it produced this SVG in **123 seconds**. [Original SVG](artwork.svg) · [Try the prompt](request.json)

![A probe follows an amber arc from the inner blue orbit to the outer one.](preview.webp)

**The physics checks out.** For orbital radii of 7,000 and 21,000 km, the two burns require **1.70 km/s** and **1.28 km/s**, with **137 minutes** of coasting. The animated path meets both orbits tangentially, and the large burn arrows point in the direction of travel. [Transfer equations](https://orbital-mechanics.space/orbital-maneuvers/hohmann-transfer.html)

Two small direction markers point backward, and Earth hides the center dot named in the caption. The SVG is the model’s unedited reply.

<details>
<summary>Run details and reproduction</summary>

Two earlier calls hit their output limits. The same prompt completed at medium reasoning effort with a 32,768-token cap. [Run record](run.json) · [Earlier outputs](attempts)

With the [local server](../../rtx-spark-setup.md) running as `flash-next`, replay the saved request from this directory.

```powershell
curl.exe -sS http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" --data-binary "@request.json" -o response.json
```

</details>
