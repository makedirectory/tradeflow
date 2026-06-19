// @ts-check
// Docusaurus site for the Alpaca Trading Engine.
// Docs are served at the site root; there are two sidebars: Usage and Engineering.

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Alpaca Trading Engine',
  tagline: 'A simple, layered, broker-agnostic algorithmic trading engine',
  favicon: 'img/favicon.ico',

  url: 'https://example.com',
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
        title: 'Alpaca Trading Engine',
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
