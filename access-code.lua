-- Set `access-code` from the project-level `access-codes` map, keyed by
-- the input file name (e.g. lec01). The map is only defined in the
-- git-ignored class profile.
function Meta(meta)
  local codes = meta["access-codes"]
  if codes == nil then return nil end
  local stem = pandoc.path.split_extension(pandoc.path.filename(quarto.doc.input_file))
  local code = codes[stem]
  if code ~= nil then
    meta["access-code"] = code
  end
  meta["access-codes"] = nil
  return meta
end
