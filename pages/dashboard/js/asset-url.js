const PLUGIN_PAGE_CONTENT_MARKER = "/api/plugin/page/content/";

function identity(url) {
  return url;
}

/**
 * 让运行时加载的本地资源继承 AstrBot Plugin Page 的短期资源令牌。
 * 仅处理当前页面同源、同插件、同 Page 作用域内的 HTTP(S) URL。
 */
export function createPluginPageAssetUrlModifier(pageHref = globalThis.location?.href) {
  if (typeof pageHref !== "string" || !pageHref) return identity;

  let pageUrl;
  try {
    pageUrl = new URL(pageHref);
  } catch {
    return identity;
  }

  const assetToken = pageUrl.searchParams.get("asset_token");
  const markerIndex = pageUrl.pathname.indexOf(PLUGIN_PAGE_CONTENT_MARKER);
  if (!assetToken || markerIndex < 0) return identity;

  const scopeStart = markerIndex + PLUGIN_PAGE_CONTENT_MARKER.length;
  const scopeParts = pageUrl.pathname.slice(scopeStart).split("/");
  if (scopeParts.length < 3 || !scopeParts[0] || !scopeParts[1]) return identity;

  const scopePath = `${pageUrl.pathname.slice(0, scopeStart)}${scopeParts[0]}/${scopeParts[1]}/`;
  return (rawUrl) => {
    if (typeof rawUrl !== "string" || !rawUrl) return rawUrl;

    let target;
    try {
      target = new URL(rawUrl, pageUrl);
    } catch {
      return rawUrl;
    }

    if (!(["http:", "https:"].includes(target.protocol)
      && target.origin === pageUrl.origin
      && target.pathname.startsWith(scopePath))) {
      return rawUrl;
    }

    target.searchParams.set("asset_token", assetToken);
    return target.href;
  };
}
