// Phase 0 shell: sidebar + main panel (§13). Controls and views arrive in Phase 3.
function App() {
  return (
    <div className="flex min-h-screen bg-neutral-950 font-mono text-neutral-200">
      <aside className="w-72 shrink-0 border-r border-neutral-800 p-4">
        <h1 className="text-sm font-semibold tracking-widest text-emerald-400 uppercase">Quant Engine</h1>
      </aside>
      <main className="flex-1 p-6 text-sm text-neutral-500">No run yet.</main>
    </div>
  )
}

export default App
