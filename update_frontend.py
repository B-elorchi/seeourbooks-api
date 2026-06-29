import pathlib

# 1. api/pipeline.ts
p = pathlib.Path('../seeourbook-client/src/api/pipeline.ts')
text = p.read_text('utf-8')
text = text.replace(
    'export async function listJobs(limit = 50, offset = 0, status?: string): Promise<PipelineJob[]> {',
    'export async function listJobs(limit = 50, offset = 0, status?: string, dateFilter?: string): Promise<PipelineJob[]> {'
)
text = text.replace(
    "if (status && status !== 'all') qs.set('status', status)",
    "if (status && status !== 'all') qs.set('status', status)\n    if (dateFilter) qs.set('date', dateFilter)"
)
p.write_text(text, 'utf-8')

# 2. hooks/usePipeline.ts
p = pathlib.Path('../seeourbook-client/src/hooks/usePipeline.ts')
text = p.read_text('utf-8')
text = text.replace(
    'export function usePipelineJobs(limit = 50, offset = 0, status?: string) {',
    'export function usePipelineJobs(limit = 50, offset = 0, status?: string, dateFilter?: string) {'
)
text = text.replace(
    "queryKey: [...pipelineKeys.jobs(), limit, offset, status ?? 'all'],",
    "queryKey: [...pipelineKeys.jobs(), limit, offset, status ?? 'all', dateFilter ?? 'none'],"
)
text = text.replace(
    "queryFn:  () => listJobs(isFiltered ? 2000 : limit, isFiltered ? 0 : offset, status),",
    "queryFn:  () => listJobs(isFiltered ? 2000 : limit, isFiltered ? 0 : offset, status, dateFilter),"
)
p.write_text(text, 'utf-8')

print("OK")
