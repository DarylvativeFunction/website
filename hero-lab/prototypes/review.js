// Claude's evaluation of each prototype (1-5 per criterion), shown in the
// lab's leaderboard under "Claude". Based on frame-by-frame gauntlet review
// at 1440x900 and 390x844. Performance is judged from element counts and
// relative software-GL cost, not from real-GPU profiling.
export const REVIEW = {
  scores: {
    A: { visual: 5, smooth: 4, read: 4, perf: 3, story: 5 },
    B: { visual: 4, smooth: 5, read: 5, perf: 4, story: 3 },
    C: { visual: 5, smooth: 4, read: 5, perf: 5, story: 3 },
    D: { visual: 4, smooth: 4, read: 4, perf: 3, story: 3 },
    E: { visual: 4, smooth: 4, read: 4, perf: 3, story: 4 },
    F: { visual: 3, smooth: 4, read: 5, perf: 5, story: 4 },
  },
  notes: {
    A: 'The only one that tells the whole story in 3D: strata on a scan line, wellbore, flow converging on the well, a grid sweeping out, points snapping to cell centres. Network phase is the generic moment; 19.8 s is the longest cycle; heaviest GPU load (44k points + bloom); portrait framing leaves empty space.',
    B: 'Smoothest and calmest; the hairline contour and grid draw-on is genuinely keynote-grade. The opening is subtle to a fault and the reservoir and ML themes are abstract, so it says less about you specifically.',
    C: 'The most distinctive: a real structure-contour map that straightens into strata and closes onto the letters as contour rings. One fullscreen shader, so the cheapest by far and razor-sharp at any size. Grid, points and network are only implied.',
    D: 'Beautiful streamline field between injector and producer wells, and the trails now write the name. The opening is the busiest frame of all six, and line count plus bloom make it one of the heavier ones.',
    E: 'Instantly reads as a reservoir simulation model; the grid reveal, exploded stack and voxel lettering are strong. More literal and "3D render" than the others.',
    F: 'The clearest data-science story (section, embedding, k-NN graph, message passing) and avoids AI-brain imagery. The opening section is dim and small, so first impressions are weaker than its middle.',
  },
  summary: 'Recommendation: C for the homepage. It is the most original, most legible and lightest, and it speaks petroleum engineering in its own visual language. If you want the full reservoir → grid → points → network narrative on screen, A is the runner-up. A strong final could graft F\'s k-NN/message-passing beat into C between "Bend" and "Contour name".',
};
