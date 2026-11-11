import json,subprocess
texts=[
"Scenario: You're training a fine-grained MoE with 16B total and 2.5B active parameters: 64 routed experts, top-6 routing, softmax gating, capacity factor 1.25, and a standard auxiliary load-balancing loss with coefficient α.",
"At α = 0.001, the busiest expert receives 9% of token assignments (uniform would be 1.56%) and 7% of tokens are dropped.",
"At α = 0.03, the busiest expert receives 1.9% and fewer than 0.5% of tokens are dropped, but validation loss is 0.04 nats worse at equal tokens.",
"At α = 0.03, the mutual information between expert selection and data domain (code, math, web) also falls by about 70%.",
"Question: Which change is most likely to keep routing balanced while recovering the lost quality?",
"A. Keep α = 0.03 but raise the capacity factor to 2.0, so no tokens are dropped and every token gets its full top-6 computation.",
"B. Go back to α = 0.001 and add a router z-loss, keeping router logits small so routing stays soft and well-spread.",
"C. Switch to top-8 routing with α = 0.01, spreading each token's load over more experts so a lower α suffices for balance.",
"D. Remove most or all of the auxiliary loss. Instead, add a per-expert bias to the router scores that is used only for top-k selection (not in the gate weights) and nudged up or down each step based on that expert's recent load.",
]
def dur(f): return float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",f]))
t=0.8; segs=[]
for i,tx in enumerate(texts):
    f=f"audio/t_k{i}.wav"; d=dur(f)
    segs.append(dict(start=round(t,3),dur=round(d,3),text=tx,file=f))
    t+=d+(1.1 if i in (3,4) else 0.75)
think=16.0
total=round(t+think,2)
json.dump(dict(segs=segs,thinkStart=round(t,3),total=total),open("timeline.json","w"),indent=1)
cmd=["ffmpeg","-y","-v","error"]
for s in segs: cmd+=["-i",s["file"]]
fl="".join(f"[{i}]adelay={int(s['start']*1000)}:all=1[a{i}];" for i,s in enumerate(segs))
fl+="".join(f"[a{i}]" for i in range(len(segs)))+f"amix=inputs={len(segs)}:normalize=0,apad,atrim=0:{total},loudnorm=I=-16:TP=-1.5[out]"
cmd+=["-filter_complex",fl,"-map","[out]","-ar","48000","-ac","2","audio/narration.wav"]
subprocess.check_call(cmd); print(total, t)
