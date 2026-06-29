import pathlib
p = pathlib.Path('../seeourbook-client/src/pages/PipelinePage.tsx')
text = p.read_text('utf-8')

# 1. State
text = text.replace(
    "const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')",
    "const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')\n  const [dateFilter, setDateFilter] = useState<string>('')"
)

# 2. isFiltered
text = text.replace(
    "const isFiltered = statusFilter !== 'all'",
    "const isFiltered = statusFilter !== 'all' || !!dateFilter"
)

# 3. Hook call
text = text.replace(
    "isFiltered ? statusFilter : undefined,\n    )",
    "isFiltered ? statusFilter : undefined,\n      dateFilter || undefined\n    )"
)

# 4. UI: Date input
# We'll put it right below the search header or next to it.
ui_date = '''          <div className="flex-1 flex flex-col pt-1">
            <h1 className="text-xl font-bold tracking-tight text-gray-900 leading-none">Pipeline</h1>
            <p className="text-xs text-gray-500 mt-1">Background AI tasks</p>
          </div>

          <div className="flex gap-2 items-center">
            <input
              type="date"
              value={dateFilter}
              onChange={e => { setDateFilter(e.target.value); setPage(0) }}
              className="text-xs border-gray-300 rounded px-2 py-1 bg-white"
            />
          </div>'''

text = text.replace(
    '''          <div className="flex-1 flex flex-col pt-1">
            <h1 className="text-xl font-bold tracking-tight text-gray-900 leading-none">Pipeline</h1>
            <p className="text-xs text-gray-500 mt-1">Background AI tasks</p>
          </div>''',
    ui_date
)

p.write_text(text, 'utf-8')
print("OK")
