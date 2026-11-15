/** `fetch`, which also fails when the server answers with an error status. */
const fetchOk = async (url: string | URL) => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return response;
};

export { fetchOk };
