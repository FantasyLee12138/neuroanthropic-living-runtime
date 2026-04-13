# AGENTS.md

## Product objective
This repository should produce frontend interfaces that are clean, production-ready, and visually coherent.
The goal is not merely to generate UI code, but to deliver interfaces with clear hierarchy, stable spacing rhythm, consistent interaction states, and minimal visual noise.

## Source of truth
- Existing design system components are the default source of truth.
- Existing theme tokens are the default source of truth for:
  - color
  - spacing
  - radius
  - typography
  - shadows
  - z-index
  - motion
- Prefer existing utilities, wrappers, hooks, layout primitives, and icons already in the repo.

## Implementation priorities
When implementing UI, optimize in this order:
1. structural correctness
2. consistency with existing design system
3. responsive behavior
4. interaction completeness
5. visual polish

## Design language
- Prefer clean visual hierarchy through spacing, typography, and grouping before using stronger color or decoration.
- Use radius conservatively. Do not exaggerate roundness unless the existing product language already does so.
- Use shadows sparingly. Avoid stacked shadows or “template-like” heavy elevation.
- Maintain a stable spacing rhythm. Prefer the repository token scale over arbitrary pixel values.
- Avoid decorative gradients unless the existing product already uses them deliberately.
- Avoid visual clutter. Remove non-essential borders, icons, labels, or containers when they do not improve comprehension.

## Typography rules
- Use the existing type scale only.
- Heading hierarchy must be obvious and stable.
- Avoid too many font sizes in one screen.
- Prefer stronger hierarchy via weight, spacing, and grouping rather than excessive color variation.
- Long text blocks must preserve readable line length and line-height.

## Component rules
- Reuse canonical components before creating new ones.
- Do not create parallel versions of common primitives such as Button, Input, Modal, Card, Tabs, Badge, Select, Table, Tooltip.
- If a new component is necessary, compose from existing primitives first.
- Preserve existing API patterns and naming conventions.

## Interaction rules
- Every interactive control should support appropriate states when relevant:
  - default
  - hover
  - focus
  - active
  - disabled
  - loading
  - error
- Focus styles must remain visible and accessible.
- Avoid layout shift during loading or async state transitions.
- Animations should be subtle and fast. Prefer clarity over flourish.

## Responsive rules
- Design mobile-first unless the route clearly targets desktop usage.
- Validate at minimum:
  - 375px
  - 768px
  - 1024px
  - 1440px
- Watch for:
  - overflow
  - awkward wrapping
  - compressed tap targets
  - inconsistent spacing after reflow
  - card density becoming too high on small screens

## Accessibility rules
- Preserve semantic HTML where possible.
- Inputs need labels or accessible equivalents.
- Icon-only buttons need accessible names.
- Contrast must remain acceptable for text and controls.
- Keyboard navigation must remain usable.

## Figma translation rules
When working from Figma:
- Treat Figma as a structural and visual reference, not as the final implementation architecture.
- Map Figma patterns into repository conventions rather than copying raw layout literally.
- Prefer token-based values over hardcoded values.
- Reuse existing variants and primitives whenever possible.

## Code quality rules
- Respect existing app architecture, routing, state management, and data-fetching patterns.
- Keep components reasonably small and composable.
- Do not introduce new dependencies unless clearly necessary.
- Do not add a new icon package if the repository already has one.
- Avoid dead code, speculative abstractions, and one-off utility sprawl.

## Definition of done
A UI task is complete only when all of the following are true:
- It matches the intended hierarchy and layout closely.
- It reuses the existing design system wherever possible.
- It behaves correctly at mobile and desktop breakpoints.
- Interactive states are present and coherent.
- It passes repository lint/typecheck/tests when applicable.
- It has been checked in a real browser, not only inferred from code.

## Expected workflow
For non-trivial UI tasks:
1. inspect existing components and tokens
2. inspect relevant route/page structure
3. implement the first pass
4. review hierarchy, spacing, and interaction quality
5. validate responsive behavior
6. refine visual polish
7. run relevant checks

## Anti-patterns to avoid
- “AI-looking” oversized radius everywhere
- excessive shadows
- arbitrary spacing values
- too many nested cards
- inconsistent typography
- overly decorative gradients
- building a second design system inside the repo
- solving visual problems by adding wrappers instead of simplifying structure