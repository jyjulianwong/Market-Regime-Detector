const HOLDINGS = [
  { ticker: 'VWRL', name: 'Vanguard FTSE All-World', value: 24810.5, change: 0.42 },
  { ticker: 'IUKD', name: 'iShares UK Dividend', value: 9120.0, change: -0.15 },
  { ticker: 'SGLN', name: 'iShares Physical Gold', value: 5400.75, change: 0.08 },
]

export default function App() {
  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">Fund Dashboard</h1>
        <p className="text-sm text-gray-500">Daily snapshot of ISA holdings</p>
      </header>

      <main className="mx-auto max-w-4xl px-6 py-10">
        <h2 className="text-lg font-semibold text-gray-800">Holdings</h2>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          {HOLDINGS.map((h) => (
            <div key={h.ticker} className="rounded-lg border border-gray-200 bg-gray-50 p-4">
              <p className="text-sm font-medium text-gray-900">{h.ticker}</p>
              <p className="text-xs text-gray-500">{h.name}</p>
              <p className="mt-2 text-xl font-semibold text-gray-900">
                £{h.value.toLocaleString('en-GB', { minimumFractionDigits: 2 })}
              </p>
              <p className={`text-xs ${h.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {h.change >= 0 ? '+' : ''}
                {h.change}%
              </p>
            </div>
          ))}
        </div>

        <a
          href="https://github.com/jyjulianwong/Investment-Fund-Reporting-AI"
          className="mt-8 inline-block rounded border border-gray-300 px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
        >
          View source
        </a>
      </main>
    </div>
  )
}
