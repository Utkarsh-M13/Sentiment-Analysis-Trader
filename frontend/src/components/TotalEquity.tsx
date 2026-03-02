const TotalEquity = () => {
  const dummyEquity = 203400;
  function formatNumber(num: number): string {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  return (
    <div className="w-40 h-30 rounded-lg purple-shadow text-white p-4 text-center">
      <div className="font-semibold text-[16px]">Total Equity</div>
      <div className="font-semibold text-2xl mt-2">${formatNumber(dummyEquity)}</div>
      <div className="text-xs mt-2 text-[#9F9F9F]">
        <span className="text-green-500 font-semibold text-sm">+2.5%</span> This Week
      </div>
    </div>
  )
}

export default TotalEquity