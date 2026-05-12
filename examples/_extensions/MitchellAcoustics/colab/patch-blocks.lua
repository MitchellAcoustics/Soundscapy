-- Helper function to merge code blocks within a div
local function merge_code_blocks(div)
    if div.classes:includes("cell") then
        local merged_code = ""
        local outputs = {}
        local current_lang = nil

        -- Iterate through div content to collect code blocks and outputs
        local i = 1
        while i <= #div.content do
            local el = div.content[i]

            -- Handle code blocks
            if el.t == "CodeBlock" then
                -- Extract language from classes
                local lang = nil
                for _, class in ipairs(el.classes) do
                    if class ~= "cell-code" then
                        lang = class
                        break
                    end
                end

                -- If language changes, create new merged block
                if current_lang and lang ~= current_lang then
                    local merged_block = pandoc.CodeBlock(merged_code)
                    merged_block.classes = {current_lang, "cell-code"}
                    table.insert(outputs, 1, merged_block)  -- Insert at beginning
                    merged_code = ""
                end

                current_lang = lang
                merged_code = merged_code .. (merged_code == "" and "" or "\n") .. el.text

                -- Remove the original code block
                table.remove(div.content, i)
                i = i - 1
            -- Handle output divs
            elseif el.t == "Div" and el.classes:includes("cell-output") then
                -- Keep output divs
                outputs[#outputs + 1] = el
                table.remove(div.content, i)
                i = i - 1
            end
            i = i + 1
        end

        -- Add final merged code block if exists
        if merged_code ~= "" then
            local merged_block = pandoc.CodeBlock(merged_code)
            merged_block.classes = {current_lang, "cell-code"}
            table.insert(outputs, 1, merged_block)  -- Insert at beginning
        end

        -- Replace div content with merged outputs
        div.content = outputs
    end
    return div
end

return {
    {
        Div = merge_code_blocks
    }
}
