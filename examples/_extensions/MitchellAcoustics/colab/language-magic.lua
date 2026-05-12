---@class SetupCell
---@field language string
---@field code string

---@class ConfigTables
---@field magicCommands table<string, string>
---@field setupCells table<string, SetupCell>

---@type table<string, string>
local DEFAULT_MAGIC_COMMANDS = {
    r = "%%R",
    julia = "%%julia",
    octave = "%%octave",
    ruby = "%%ruby",
    perl = "%%perl",
    bash = "%%bash",
    sh = "%%bash",
    sql = "%%sql",
    sas = "%%sas"
}

---@type table<string, SetupCell>
local DEFAULT_SETUP_CELLS = {
    r = {
        language = "python",
        code = [[
# Install and configure R integration
!pip install rpy2
%load_ext rpy2.ipython]]
    },
    octave = {
        language = "python",
        code = [[
# Install and configure Octave integration
!apt-get install -y octave octave-signal
!pip install oct2py octave_kernel metakernel
%load_ext oct2py.ipython]]
    },
    sql = {
        language = "python",
        code = [[
# Install and configure SQL magic
!pip install jupysql duckdb-engine
%load_ext sql
%config SqlMagic.autopandas = True
%config SqlMagic.autopolars = True
%config SqlMagic.feedback = 0
%config SqlMagic.displaylimit = 10]]
    },
    sas = {
        language = "python",
        code = [[
# This requires a licensed copy of SAS
# For details on SASPy, see: https://support.sas.com/ondemand/saspy.html
# For Colab, you can use local runtimes to connect to a local SAS installation
# https://research.google.com/colaboratory/local-runtimes.html
# Install and configure SAS integration
!pip install saspy sas_kernel
%load_ext sas_magic]]
    },
    julia = {
        language = "python",
        code = [[
# Install and configure Julia integration
!pip install -qqq juliacall
# Load Julia extension
from juliacall import Main as jl
# Load Julia magic
%load_ext juliacall.ipython
    ]]}
}

-- Global variables to store configuration
---@type table<string, string>
local magicCommands = {}
---@type table<string, SetupCell>
local setupCells = {}

---@param t1 table
---@param t2 table
---@return table
local function merge_tables(t1, t2)
    local result = {}
    for k, v in pairs(t1) do result[k] = v end
    for k, v in pairs(t2) do result[k] = v end
    return result
end

---@param text string
---@return string
local function normalize_quotes(text)
    -- Keep code ASCII-safe if smart punctuation is introduced during metadata parsing.
    local normalized = text
        :gsub("“", '"')
        :gsub("”", '"')
        :gsub("‘", "'")
        :gsub("’", "'")
    return normalized
end

---@param inlines pandoc.Inlines
---@return string[]
local function inlines_to_lines(inlines)
    local lines = {}
    local parts = {}

    for _, inline in ipairs(inlines) do
        if inline.t == "LineBreak" or inline.t == "SoftBreak" then
            if #parts > 0 then
                table.insert(lines, normalize_quotes(pandoc.utils.stringify(parts)))
                parts = {}
            end
        else
            table.insert(parts, inline)
        end
    end

    if #parts > 0 then
        table.insert(lines, normalize_quotes(pandoc.utils.stringify(parts)))
    end

    return lines
end

---@param value any
---@return string
local function meta_value_to_code(value)
    local ptype = pandoc.utils.type(value)

    if ptype == "Blocks" then
        local lines = {}
        for _, block in ipairs(value) do
            if block.t == "Header" then
                local prefix = string.rep("#", block.level) .. " "
                table.insert(lines, prefix .. normalize_quotes(pandoc.utils.stringify(block.content)))
            elseif block.t == "Para" or block.t == "Plain" then
                for _, line in ipairs(inlines_to_lines(block.content)) do
                    table.insert(lines, line)
                end
            elseif block.t == "CodeBlock" or block.t == "RawBlock" then
                table.insert(lines, normalize_quotes(block.text))
            else
                table.insert(lines, normalize_quotes(pandoc.utils.stringify(block)))
            end
        end
        return table.concat(lines, "\n")
    end

    return normalize_quotes(pandoc.utils.stringify(value))
end

---@param block pandoc.CodeBlock
---@return boolean
local function isCellEvaluated(block)
    for _, class in ipairs(block.attr.classes) do
        if class == "cell-code" then
            return true
        end
    end
    return false
end

---@param meta pandoc.Meta
---@return pandoc.Meta
function Meta(meta)
    magicCommands = merge_tables(DEFAULT_MAGIC_COMMANDS, meta['magic-commands'] or {})

    -- Convert user-defined setup-cells from Pandoc MetaMap to plain Lua tables
    local userSetupCells = {}
    if meta['setup-cells'] then
        for lang, setup in pairs(meta['setup-cells']) do
            if setup and setup.code then
                userSetupCells[lang] = {
                    language = setup.language and pandoc.utils.stringify(setup.language) or lang,
                    code = setup.code and meta_value_to_code(setup.code) or ""
                }
            end
        end
    end
    setupCells = merge_tables(DEFAULT_SETUP_CELLS, userSetupCells)
    return meta
end

---@param doc pandoc.Doc
---@return table<string, boolean>
function detectLanguages(doc)
    ---@type table<string, boolean>
    local languages = {}

    local function traverse(blocks)
        for _, block in pairs(blocks) do
            if block.t == "CodeBlock" then
                -- Only count evaluated cells (those with "cell-code" class)
                if isCellEvaluated(block) then
                    local lang = block.attr.classes[1]
                    if lang and lang ~= "cell-code" then
                        languages[lang] = true
                    end
                end
            elseif block.t == "Div" then
                traverse(block.content)
            end
        end
    end

    traverse(doc.blocks)
    return languages
end

---@param detectedLangs table<string, boolean>
---@return pandoc.Div[]
function createSetupCells(detectedLangs)
    ---@type pandoc.Div[]
    local cells = {}

    for lang, _ in pairs(detectedLangs) do
        if setupCells[lang] then
            local setup = setupCells[lang]
            local cell = pandoc.Div({pandoc.CodeBlock(
                setup.code,
                pandoc.Attr("", {setup.language, "cell-code"})
            )},
            pandoc.Attr("", {"cell"}))
            table.insert(cells, cell)
        end
    end

    return cells
end

---@param block pandoc.CodeBlock
---@return pandoc.CodeBlock
function CodeBlock(block)
    if not isCellEvaluated(block) then
        return block
    end

    local lang = block.attr.classes[1]
    if lang and magicCommands[lang] then
        block.text = magicCommands[lang] .. "\n" .. block.text
        return block
    end
    return block
end

---@param doc pandoc.Doc
---@return pandoc.Doc
function Pandoc(doc)
    if not quarto.doc.is_format("ipynb") then return doc end

    local detectedLangs = detectLanguages(doc)
    local setupBlocks = createSetupCells(detectedLangs)

    for i = #setupBlocks, 1, -1 do
        table.insert(doc.blocks, 1, setupBlocks[i])
    end

    return doc
end

return {
    { Meta = Meta },
    { CodeBlock = CodeBlock },
    { Pandoc = Pandoc }
}
