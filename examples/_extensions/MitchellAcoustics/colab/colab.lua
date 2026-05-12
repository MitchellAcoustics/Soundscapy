-- Type definition for GitHub information
---@class GitHubInfo
---@field user string
---@field repo string
---@field branch string
---@field scrollTo string|nil

-- Function to extract GitHub information from metadata
---@param meta pandoc.Meta
---@return GitHubInfo|nil
local function extract_github_info(meta)

    -- Check if the 'colab' field exists in the metadata
    if not meta.colab then
        return nil
    end

    -- Extract GitHub user, repo, and branch from metadata
    ---@type string
    local github_user = pandoc.utils.stringify(meta.colab["gh-user"])
    ---@type string
    local github_repo = pandoc.utils.stringify(meta.colab["gh-repo"])
    ---@type string
    local github_branch = pandoc.utils.stringify(meta.colab["gh-branch"]) or "main" -- Default to "main" if not specified
    ---@type string|nil
    local scroll_to = meta.colab["scroll-to"] and pandoc.utils.stringify(meta.colab["scroll-to"]) or nil

    -- Validate required fields
    if not (github_user and github_repo) then
        quarto.log.error("Warning: Missing required Colab metadata fields (gh-user or gh-repo).\n")
        return nil
    end

    -- Return GitHub information as a table
    return {
        user = github_user,
        repo = github_repo,
        branch = github_branch,
        scrollTo = scroll_to
    }
end

-- Function to create Colab badge
---@return pandoc.Image
local function create_colab_badge()
    return pandoc.Image("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg")
end

-- Function to construct Colab URL that links to
---@param github_info GitHubInfo
---@param notebook_relative_path string
---@return string
local function construct_colab_url(github_info, notebook_relative_path)
    --- https://colab.research.google.com/github/<gh-username>/<repo>/blob/<branch>/<path>
    local base_url = string.format(
        "https://colab.research.google.com/github/%s/%s/blob/%s/%s",
        github_info.user,
        github_info.repo,
        github_info.branch,
        notebook_relative_path
    )

    -- Append scrollTo parameter if provided
    if github_info.scrollTo then
        return base_url .. "#scrollTo=" .. github_info.scrollTo
    else
        return base_url
    end
end

-- Main filter function applied to the Pandoc document to insert the Colab link
---@param doc pandoc.Pandoc
---@return pandoc.Pandoc
function Pandoc(doc)

    -- Check if the document is a Jupyter notebook
    if not quarto.doc.is_format("ipynb") then return doc end

    -- Extract GitHub information from metadata
    local github_info = extract_github_info(doc.meta)

    if github_info then
        -- Get the notebook name from the metadata or use a default
        ---@type string | nil
        local notebook_name = quarto.doc.project_output_file() or quarto.doc.output_file

        -- Construct the Colab URL
        ---@type string
        local colab_url = construct_colab_url(github_info, notebook_name)

        -- Create the Colab badge link
        ---@type pandoc.Image
        local colab_badge = create_colab_badge()
        ---@type pandoc.Link
        local colab_link = pandoc.Link(colab_badge, colab_url)

        -- Create a new block with the Colab link
        ---@type pandoc.Block
        local colab_block = pandoc.Para({colab_link})

        -- Insert the Colab link block at the beginning of the document
        table.insert(doc.blocks, 1, colab_block)
    end

    return doc
end
