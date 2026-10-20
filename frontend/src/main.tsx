import { StrictMode, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ErrorBox } from './components/ui'
import { DEMO, loadManifest } from './demo'

const root = createRoot(document.getElementById('root')!)

function render(node: ReactNode) {
  root.render(<StrictMode>{node}</StrictMode>)
}

if (DEMO) {
  // The static demo (Phase 9) needs its manifest before the first render: its scenarios are the
  // sidebar's defaults.
  loadManifest().then(
    (manifest) => render(<App demo={manifest} />),
    (error: unknown) =>
      render(
        <div className="p-6 font-mono">
          <ErrorBox title="The demo could not start" detail={error instanceof Error ? error.message : String(error)} />
        </div>,
      ),
  )
} else {
  render(<App />)
}
