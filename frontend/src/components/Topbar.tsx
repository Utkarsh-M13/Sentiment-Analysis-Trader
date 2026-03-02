const Topbar = () => {
  return (
    <div className="w-full h-20 bg-[#11101A] fixed top-0 border-b border-[#2D2C3A] pl-8 pr-8 flex items-center justify-between z-50">
      <div>
            <div className="text-white font-semibold text-[20px]">
              Sentiment Analysis Trader
            </div>
            <div className='bg-gradient bg-clip-text text-transparent'>
              Utkarsh Majithia
            </div>
      </div>
      <div className='cursor-pointer bg-white rounded-xl hidden lg:inline'>
        <a href="https://www.utkarsh-dev.com/">
        <img className='rounded-xl w-12 h-12 lg:w-16 lg:h-16 border-white border' src="/assets/SelfPortrait.svg" alt="" />
        </a>
          
      </div>
    </div>  
  )
}

export default Topbar