# Tissue box texture atlas (Monalisa 250)

## Files
- `atlas.png` 6904x3464, 10 px/mm, cross net, 32 px margin, 12 px edge bleed. Clean texture (no labels).
- `atlas_layout.json` per-face pixel rect, UVs (OBJ convention, origin bottom-left), 3D corner coords, source photo.
- `faces/*.png` individual rectified faces, already in net orientation.
- `box.obj` + `box.mtl` textured cuboid, 24 verts (4 per face), meters, outward CCW winding.
- `atlas_labeled.jpg`, `preview_3d.jpg` for human verification only.
- `build_atlas.py`, `config.json` reproducible pipeline. `python3 build_atlas.py config.json <photo_dir> <out_dir>`

## Conventions
Net (each face seen from outside):

            [ top  ]
    [left ] [front ] [right] [back ]
            [bottom]

Frame: origin = box center, +X = right (L), +Y = up (H), +Z = front (W).
front = lavender face with the tissue-level window. left = -X = on the viewer's left when facing front.

## Known assumptions / unverified
1. **Dimensions are estimated, not measured.** Ratio L:W:H = 1.98 : 0.97 : 1 was recovered from perspective (EXIF f35=21mm); the three independent ratios agree within ~1.5%. Absolute scale L=230 mm is a guess. Measure with a ruler, edit `dims_mm` in config.json, re-run.
2. **left.jpg / right.jpg are swapped relative to the slots.** Evidence: the dermatest logo on the top face is visible at the near-right of left.jpg, and the glossy patch at the opposite top corner is visible at the near-left of front.jpg. That puts left.jpg's face at +X. To revert, swap the `photo` fields of left/right in config.json.
3. **Top rotation** follows from the same evidence (logo ends up at back-right, text upright from the front).
4. **Bottom rotation is NOT verified from the photos.** Assumed text reads upright when the box is tipped forward (text-up toward the front edge). If wrong, set bottom `rot_ccw90` to 3.
5. Lighting is baked in as photographed (hand/phone shadow on top and bottom, specular patch on top, per-photo exposure differences). No de-shading applied.
6. Corner accuracy roughly 5-10 px (0.5-1 mm). 6 px inset applied to avoid background slivers.
