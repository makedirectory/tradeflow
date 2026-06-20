// @ts-check
// Docusaurus site for TradeFlow.
// Docs are served at the site root; there are two sidebars: Usage and Engineering.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'TradeFlow',
  tagline: 'A simple, layered, broker-agnostic algorithmic trading engine',
  favicon: 'img/favicon.ico',

  // Served as its own site at the docs subdomain. `baseUrl` stays '/' because the
  // docs own the whole (sub)domain root; it would only change if these were ever
  // hosted under a path like mk-dir.com/docs.
  // Note: on S3 + CloudFront set `trailingSlash: true` and add a CloudFront
  // Function to append `index.html` for directory URLs; Vercel/Cloudflare Pages
  // handle clean URLs with the default.
  url: 'https://tradeflow.mk-dir.com',
  baseUrl: '/',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: '/', // docs at site root
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'TradeFlow',
        items: [
          { type: 'docSidebar', sidebarId: 'usage', position: 'left', label: 'Usage' },
          { type: 'docSidebar', sidebarId: 'engineering', position: 'left', label: 'Engineering' },
        ],
      },
      footer: {
        style: 'dark',
        copyright: 'Educational software. Trade at your own risk.',
      },
      prism: {
        additionalLanguages: ['bash', 'python', 'toml'],
      },
    }),
};

module.exports = config;
