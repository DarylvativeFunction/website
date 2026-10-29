// The six hero prototypes. Each module default-exports a prototype definition
// (see README.md, "Prototype contract"). Loaded lazily so one failing
// prototype never takes the lab down.

export const PROTOTYPES = [
  { id: 'A', name: 'Cinematic Reservoir', short: 'Reservoir', load: () => import('./a-reservoir.js') },
  { id: 'B', name: 'Minimal Premium', short: 'Minimal', load: () => import('./b-minimal.js') },
  { id: 'C', name: 'Contour-Line Reveal', short: 'Contour', load: () => import('./c-contour.js') },
  { id: 'D', name: 'Flow-Path Reveal', short: 'Flow-Path', load: () => import('./d-flowpath.js') },
  { id: 'E', name: 'Voxel Collapse', short: 'Voxel', load: () => import('./e-voxel.js') },
  { id: 'F', name: 'Graph Learning', short: 'Graph', load: () => import('./f-graph.js') },
];
