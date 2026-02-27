import { defineConfig } from 'vitepress'
import fs from 'node:fs'
import path from 'node:path'

/** 递归统计目录下的论文 .md 文件数（排除 README.md / index.md） */
function countPapers(dir: string): number {
  if (!fs.existsSync(dir)) return 0
  let count = 0
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      count += countPapers(path.join(dir, entry.name))
    } else if (
      entry.name.endsWith('.md') &&
      entry.name !== 'README.md' &&
      entry.name !== 'index.md'
    ) {
      count++
    }
  }
  return count
}

export default defineConfig({
  title: 'LLM Paper Notes',
  description: 'LLM 及相关领域论文精读笔记',
  lang: 'zh-CN',
  base: '/',

  ignoreDeadLinks: true,

  transformPageData(pageData, { siteConfig }) {
    if (pageData.frontmatter.layout === 'home' && pageData.frontmatter.features) {
      for (const feature of pageData.frontmatter.features) {
        if (feature.link?.startsWith('/papers/')) {
          const dir = path.join(siteConfig.srcDir, feature.link.slice(1))
          const count = countPapers(dir)
          if (count > 0) {
            feature.linkText = `已有 ${count} 篇笔记`
          }
        }
      }
    }
  },

  rewrites: {
    'papers/:dir/README.md': 'papers/:dir/index.md',
  },

  markdown: {
    math: true,
  },

  head: [
    ['link', { rel: 'icon', href: 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🧠</text></svg>' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { href: 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap', rel: 'stylesheet' }],
  ],

  themeConfig: {
    logo: undefined,
    siteTitle: '🧠 LLM Paper Notes',

    nav: [
      { text: '首页', link: '/' },
      { text: '论文索引', link: '/papers/' },
      { text: '笔记模板', link: '/templates/paper_template' },
    ],

    sidebar: {
      '/papers/': [
        {
          text: '📚 论文分类',
          items: [
            {
              text: '01 Foundation Models',
              collapsed: true,
              link: '/papers/01-foundation-models/',
              items: [],
            },
            {
              text: '02 Alignment & Safety',
              collapsed: true,
              link: '/papers/02-alignment-and-safety/',
              items: [],
            },
            {
              text: '03 Reasoning',
              collapsed: true,
              link: '/papers/03-reasoning/',
              items: [],
            },
            {
              text: '04 Multimodal',
              collapsed: true,
              link: '/papers/04-multimodal/',
              items: [],
            },
            {
              text: '05 Agents',
              collapsed: true,
              link: '/papers/05-agents/',
              items: [],
            },
            {
              text: '06 Embodied AI',
              collapsed: false,
              link: '/papers/06-embodied-ai/',
              items: [
                {
                  text: 'VLA',
                  collapsed: false,
                  items: [
                    {
                      text: '基础模型',
                      collapsed: false,
                      items: [
                        { text: 'π₀ (2024)', link: '/papers/06-embodied-ai/vla/foundation/pi0_2024' },
                        { text: 'π₀.₅ (2025)', link: '/papers/06-embodied-ai/vla/foundation/pi05_2025' },
                      ],
                    },
                    {
                      text: 'RL 后训练',
                      collapsed: false,
                      items: [
                        { text: 'RISE (2026)', link: '/papers/06-embodied-ai/vla/rl/RISE_2026' },
                        { text: 'RLinf (2025)', link: '/papers/06-embodied-ai/vla/rl/RLinf_2025' },
                        { text: 'SAC Flow (2026)', link: '/papers/06-embodied-ai/vla/rl/SAC_Flow_2026' },
                        { text: 'VLA-RL (2025)', link: '/papers/06-embodied-ai/vla/rl/VLA_RL_2025' },
                        { text: 'WoVR (2026)', link: '/papers/06-embodied-ai/vla/rl/WoVR_2026' },
                      ],
                    },
                  ],
                },
                {
                  text: 'World Models',
                  collapsed: true,
                  items: [],
                },
              ],
            },
            {
              text: '07 Efficiency',
              collapsed: true,
              link: '/papers/07-efficiency/',
              items: [],
            },
            {
              text: '08 RAG & Knowledge',
              collapsed: true,
              link: '/papers/08-rag-and-knowledge/',
              items: [],
            },
            {
              text: '09 Evaluation',
              collapsed: true,
              link: '/papers/09-evaluation-and-benchmarks/',
              items: [],
            },
          ],
        },
      ],
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索',
            buttonAriaLabel: '搜索',
          },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清除搜索',
            footer: {
              selectText: '选择',
              navigateText: '导航',
              closeText: '关闭',
            },
          },
        },
      },
    },

    outline: {
      level: [2, 3],
      label: '目录',
    },

    darkModeSwitchLabel: '主题',
    returnToTopLabel: '回到顶部',
    lastUpdated: {
      text: '最后更新',
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/jiabingyang01/llm-paper-notes' },
    ],

    footer: {
      message: '基于 CC BY-SA 4.0 协议',
      copyright: '© <a href="https://github.com/jiabingyang01" target="_blank">jiabingyang01</a>',
    },
  },
})
