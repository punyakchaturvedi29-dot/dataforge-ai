import '@testing-library/jest-dom/vitest'
// recharts' ResponsiveContainer needs ResizeObserver in jsdom
globalThis.ResizeObserver = globalThis.ResizeObserver || class { observe() {} unobserve() {} disconnect() {} }
window.matchMedia = window.matchMedia || (() => ({ matches: false, addEventListener() {}, removeEventListener() {} }))
// vitest runs without globals, so Testing Library cannot register its own cleanup
import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'
afterEach(() => cleanup())
